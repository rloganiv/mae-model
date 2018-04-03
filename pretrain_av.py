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
architecture.
"""
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
               config['model']['embedding_size']),
        initializer=tf.random_uniform_initializer(-1.0 / 200, 1.0 / 200))
    value_embeddings = tf.get_variable(
        'value_embeddings',
        dtype=tf.float32,
        shape=(config['data']['value_vocab_size'],
               config['model']['embedding_size']),
        initializer=tf.random_uniform_initializer(-1.0 / 200, 1.0 / 200))

    # Used by model / loss function.
    attr_queries = tf.nn.embedding_lookup(attr_embeddings, attr_query_ids)
    correct_values = tf.nn.embedding_lookup(value_embeddings, correct_value_ids)
    incorrect_values = tf.nn.embedding_lookup(value_embeddings, incorrect_value_ids)

    distance_metric=config['model']['distance_metric']
    s = utils.distance(attr_queries, correct_values, distance_metric)
    s_prime = utils.distance(attr_queries, incorrect_values, distance_metric)
    loss = tf.maximum(0.0, 1 + s - s_prime)
    loss = tf.reduce_mean(loss)
    tf.losses.add_loss(loss)

    # Summaries
    unk = tf.equal(correct_value_ids, config['data']['value_vocab_size'] - 1)
    obs = tf.logical_not(unk)

    mean_loss, _ = tf.metrics.mean(loss,
                                   metrics_collections=['metrics'],
                                   updates_collections=['updates'],
                                   name='streaming_loss')
    tf.summary.scalar('loss', mean_loss)

    scores = utils.distance_matrix(attr_queries, value_embeddings, distance_metric)
    rank = utils.rank(scores, correct_value_ids)

    mrr, _ = tf.metrics.mean(1.0 / rank,
                             metrics_collections=['metrics'],
                             updates_collections=['updates'],
                             name='streaming_mrr')
    mrr_obs, _ = tf.metrics.mean(1.0 / tf.boolean_mask(rank, obs),
                                 metrics_collections=['metrics'],
                                 updates_collections=['updates'],
                                 name='streaming_mrr_obs')
    mrr_unk, _ = tf.metrics.mean(1.0 / tf.boolean_mask(rank, unk),
                                 metrics_collections=['metrics'],
                                 updates_collections=['updates'],
                                 name='streaming_mrr_unk')
    tf.summary.scalar('mean_reciprocal_rank', mrr)
    tf.summary.scalar('mean_reciprocal_rank_obs', mrr_obs)
    tf.summary.scalar('mean_reciprocal_rank_unk', mrr_unk)

    lt_1 = tf.cast(rank <= 1.0, dtype=tf.float32)
    lt_20 = tf.cast(rank <= 20.0, dtype=tf.float32)
    acc_at_1, _ = tf.metrics.mean(lt_1,
                                  metrics_collections=['metrics'],
                                  updates_collections=['updates'],
                                  name='streaming_acc_at_1')
    acc_at_1_obs, _ = tf.metrics.mean(tf.boolean_mask(lt_1, obs),
                                      metrics_collections=['metrics'],
                                      updates_collections=['updates'],
                                      name='streaming_acc_at_1_obs')
    acc_at_1_unk, _ = tf.metrics.mean(tf.boolean_mask(lt_1, unk),
                                      metrics_collections=['metrics'],
                                      updates_collections=['updates'],
                                      name='streaming_acc_at_1_unk')
    acc_at_20, _ = tf.metrics.mean(lt_20,
                                  metrics_collections=['metrics'],
                                  updates_collections=['updates'],
                                  name='streaming_acc_at_20')
    acc_at_20_obs, _ = tf.metrics.mean(tf.boolean_mask(lt_20, obs),
                                      metrics_collections=['metrics'],
                                      updates_collections=['updates'],
                                      name='streaming_acc_at_20_obs')
    acc_at_20_unk, _ = tf.metrics.mean(tf.boolean_mask(lt_20, unk),
                                      metrics_collections=['metrics'],
                                      updates_collections=['updates'],
                                      name='streaming_acc_at_20_unk')
    tf.summary.scalar('accuracy_at_1', acc_at_1)
    tf.summary.scalar('accuracy_at_1_obs', acc_at_1_obs)
    tf.summary.scalar('accuracy_at_1_unk', acc_at_1_unk)
    tf.summary.scalar('accuracy_at_20', acc_at_20)
    tf.summary.scalar('accuracy_at_20_obs', acc_at_20_obs)
    tf.summary.scalar('accuracy_at_20_unk', acc_at_20_unk)


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

            # Evaluate on test data.
            for batch in utils.generate_batches('val', config):
                sess.run(update_op, feed_dict=batch)
            print(sess.run(metric_op))

            # Write summaries.
            summary = sess.run(summary_op, feed_dict=batch)
            eval_logger.add_summary(summary, 0)

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

