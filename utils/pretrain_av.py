# Copyright 2018 The MAE Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learns attribute and value embeddings which help warm-start the MAE
architecture."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf

import utils

slim = tf.contrib.slim


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('config', '',
                       'Configuration file.')

tf.logging.set_verbosity(tf.logging.INFO)


def cosine_similarity(x1, x2, pairwise=False):
    x1 = tf.nn.l2_normalize(x1, dim=1)
    x2 = tf.nn.l2_normalize(x2, dim=1)
    if pairwise:
        return tf.tensordot(x1, x2, axes=[[1], [1]])
    else:
        return tf.reduce_sum(x1 * x2, axis=1)


def rank_op(scores, correct_value_ids):
    batch_size = correct_value_ids.shape[0]
    indices = tf.stack([tf.range(batch_size), correct_value_ids], axis=1)
    correct = tf.gather_nd(scores, indices)
    correct = tf.expand_dims(correct, 1)
    incorrect = tf.cast(scores > correct, dtype=tf.float32)
    rank = tf.reduce_sum(incorrect, axis=1) + 1.0
    return(rank)


def build_graph(config):
    batch_size = config['training']['batch_size']

    attr_query_ids = tf.placeholder(tf.int32, shape=(batch_size,),
                                    name='attr_query_ids')
    correct_value_ids = tf.placeholder(tf.int32, shape=(batch_size,),
                                       name='correct_value_ids')
    incorrect_value_ids = tf.placeholder(tf.int32, shape=(batch_size,),
                                         name='incorrect_value_ids')

    # Embedding matrices.
    attr_embeddings = tf.get_variable(
        'attr_embeddings',
        dtype=tf.float32,
        shape=(config['data']['attr_vocab_size'],
               config['model']['embedding_size']))
    value_embeddings = tf.get_variable(
        'value_embeddings',
        dtype=tf.float32,
        shape=(config['data']['value_vocab_size'],
               config['model']['embedding_size']))

    # Used by model / loss function.
    attr_queries = tf.nn.embedding_lookup(attr_embeddings, attr_query_ids)
    correct_values = tf.nn.embedding_lookup(value_embeddings, correct_value_ids)
    incorrect_values = tf.nn.embedding_lookup(value_embeddings, incorrect_value_ids)

    s = 1 - cosine_similarity(attr_queries, correct_values)
    s_prime = tf.maximum(0.0, cosine_similarity(attr_queries, incorrect_values))
    loss = s + s_prime
    loss = tf.reduce_mean(loss)
    tf.losses.add_loss(loss)

    # Summaries
    mean_loss, _ = tf.metrics.mean(loss,
                                   metrics_collections=['metrics'],
                                   updates_collections=['updates'],
                                   name='streaming_loss')
    tf.summary.scalar('loss', mean_loss)

    scores = cosine_similarity(attr_queries, value_embeddings, pairwise=True)
    rank = rank_op(scores, correct_value_ids)

    mrr, _ = tf.metrics.mean(1.0 / rank,
                             metrics_collections=['metrics'],
                             updates_collections=['updates'],
                             name='streaming_mrr')
    tf.summary.scalar('mean_reciprocal_rank', mrr)

    lt_1 = tf.cast(rank <= 1.0, dtype=tf.float32)
    lt_20 = tf.cast(rank <= 20.0, dtype=tf.float32)
    acc_at_1, _ = tf.metrics.mean(lt_1,
                                  metrics_collections=['metrics'],
                                  updates_collections=['updates'],
                                  name='streaming_acc_at_1')
    acc_at_20, _ = tf.metrics.mean(lt_20,
                                   metrics_collections=['metrics'],
                                   updates_collections=['updates'],
                                   name='streaming_acc_at_20')
    tf.summary.scalar('accuracy_at_1', acc_at_1)
    tf.summary.scalar('accuracy_at_20', acc_at_20)


def get_init_fn(config):
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])

    # Try to load existing checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    global_saver = tf.train.Saver()
    if latest_checkpoint:
        tf.logging.info('Found existing checkpoint: %s' % latest_checkpoint)
        def init_fn(sess):
            return global_saver.restore(sess, latest_checkpoint)
        return init_fn
    else:
        tf.logging.info('No existing checkpoint found')
        def init_fn(sess):
            pass
        return init_fn


def main(_):
    # Load config
    config = utils.load_config(FLAGS.config)

    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])
    ckpt = os.path.join(ckpt_dir, 'av.ckpt')

    log_dir = os.path.join(config['training']['log_dir'],
                           config['experiment_name'])
    if not os.path.exists(ckpt_dir):
        tf.logging.info('Creating checkpoint directory: %s' % ckpt_dir)
        os.mkdir(ckpt_dir)
    if not os.path.exists(log_dir):
        tf.logging.info('Creating log directory: %s' % log_dir)


    g = tf.Graph()
    with g.as_default():
        tf.logging.info('Creating graph')
        build_graph(config)
        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver(max_to_keep=5)

        init_fn = get_init_fn(config)

        total_loss = tf.losses.get_total_loss()
        optimizer = tf.train.AdamOptimizer(config['training']['learning_rate'])
        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 clip_gradient_norm=config['training']['gradient_clipping'],
                                                 summarize_gradients=True)
        summary_op = tf.summary.merge_all()
        eval_logger = tf.summary.FileWriter(log_dir)

        metric_op = tf.get_collection('metrics')
        update_op = tf.get_collection('updates')

        streaming_vars = [i for i in tf.local_variables() if 'streaming' in i.name]
        reset_op = [tf.variables_initializer(streaming_vars)]

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            init_fn(sess) # Responsible for restoring variables / warm starts

            # Generate data loop.
            for batch in utils.generate_batches('train', config):

                try:
                    i, _ = sess.run([global_step, train_op], feed_dict=batch)
                except tf.errors.InvalidArgumentError: # Encountered a bad JPEG
                    continue

                if not i % config['training']['log_frequency']:
                    loss = sess.run(total_loss, feed_dict=batch)
                    tf.logging.info('Iteration %i - Loss: %0.4f' % (i, loss))

                if not i % config['training']['save_frequency']:
                    tf.logging.info('Saving checkpoint for iteration %i' % i)
                    saver.save(sess, ckpt)

                    sess.run(reset_op)
                    # Evaluate on test data.
                    for batch in utils.generate_batches('val', config):
                        sess.run(update_op, feed_dict=batch)
                    print(sess.run(metric_op))

                    # Write summaries.
                    summary = sess.run(summary_op, feed_dict=batch)
                    eval_logger.add_summary(summary, i)

                if i >= config['training']['max_steps']:
                    tf.logging.info('Training complete')
                    sys.exit(0)


if __name__ == '__main__':
    tf.app.run()

