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
"""MAE model training script.

Usage:
    python train.py --config config.yaml
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys
import tensorflow as tf
import warnings

from nets.mae import mae
import utils

slim = tf.contrib.slim


FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def preprocess_image_byte_strings(image_byte_strings):
    """Preprocesses the image byte string tensors.

    Args:
        image_byte_strings: A tensor of size [batch_size, n_images].
    Returns:
        A tensor of size [batch_size, n_images, 224, 224, 3] containing the
            image data.
    """
    # The processing applied to each individual image.
    def _subroutine(x):
        x = tf.image.decode_jpeg(x, channels=3)
        x = utils.preprocess_image(x, output_height=224, output_width=224)
        return x
    # tf.map_fn will allow us to apply _subroutine() to sequences of images,
    # however there is a slight complication in that there are two sequence
    # dimensions - one for the batch, and the other for the sequence of images
    # for a given item. In order to fix this we use the same trick as we do to
    # make VGG net work - namely, we combine these dimensions before feeding
    # and then reseparate them when we are done.
    batch_size = image_byte_strings.shape[0]
    x = tf.reshape(image_byte_strings, [-1])
    x = tf.map_fn(_subroutine, x, dtype=tf.float32)
    x = tf.reshape(x, shape=(batch_size, -1, 224, 224, 3))
    return x


def build_graph(config):
    batch_size = config['training']['batch_size']

    # === Required Inputs ===

    # Placeholders.
    attr_query_ids = tf.placeholder(tf.int32, shape=(batch_size,),
                                    name='attr_query_ids')
    correct_value_ids = tf.placeholder(tf.int32, shape=(batch_size,),
                                       name='correct_value_ids')
    incorrect_value_ids = tf.placeholder(tf.int32, shape=(batch_size,),
                                         name='incorrect_value_ids')

    # Embedding matrices.
    embedding_size = config['model']['context_embedding_size']
    attr_embeddings = tf.get_variable(
        'attr_embeddings',
        dtype=tf.float32,
        shape=(config['data']['attr_vocab_size'], embedding_size),
        trainable=config['model']['trainable_attr_embeddings'],
        initializer=tf.random_uniform_initializer(-1.0 / embedding_size,
                                                   1.0 / embedding_size))
    value_embeddings = tf.get_variable(
        'value_embeddings',
        dtype=tf.float32,
        shape=(config['data']['value_vocab_size'], embedding_size),
        trainable=config['model']['trainable_value_embeddings'],
        initializer=tf.random_uniform_initializer(-1.0 / embedding_size,
                                                   1.0 / embedding_size))

    # Used by model / loss function.
    attr_queries = tf.nn.embedding_lookup(attr_embeddings, attr_query_ids)
    correct_values = tf.nn.embedding_lookup(value_embeddings,
                                            correct_value_ids)
    incorrect_values = tf.nn.embedding_lookup(value_embeddings,
                                              incorrect_value_ids)

    # === Optional Inputs ===

    # Descriptions.
    if config['model']['use_descs']:
        desc_word_ids = tf.placeholder(tf.int32, shape=(batch_size, None),
                                       name='desc_word_ids')
        desc_word_embeddings = tf.get_variable(
            'desc_word_embeddings',
            dtype=tf.float32,
            shape=(config['data']['desc_vocab_size'],
                   config['model']['word_embedding_size']),
            trainable=config['model']['trainable_word_embeddings'])
        desc_encoder_inputs = tf.nn.embedding_lookup(desc_word_embeddings,
                                                     desc_word_ids)
        desc_encoder_masks = tf.placeholder(tf.float32, shape=(batch_size, None),
                                            name='desc_masks')
        desc_encoder_params = config['model']['desc_encoder_params']
    else:
        desc_encoder_inputs = None
        desc_encoder_masks = None
        desc_encoder_params = {}

    # Titles.
    if config['model']['use_titles']:
        title_word_ids = tf.placeholder(tf.int32, shape=(batch_size, None),
                                       name='title_word_ids')
        title_word_embeddings = tf.get_variable(
            'title_word_embeddings',
            dtype=tf.float32,
            shape=(config['data']['desc_vocab_size'],
                   config['model']['word_embedding_size']),
            trainable=config['model']['trainable_word_embeddings'])
        title_encoder_inputs = tf.nn.embedding_lookup(title_word_embeddings,
                                                     title_word_ids)
        title_encoder_masks = tf.placeholder(tf.float32, shape=(batch_size, None),
                                            name='title_masks')
        title_encoder_params = config['model']['title_encoder_params']
    else:
        title_encoder_inputs = None
        title_encoder_masks = None
        title_encoder_params = {}

    # Images.
    if config['model']['use_images']:
        image_byte_strings = tf.placeholder(tf.string, shape=(batch_size, None),
                                            name='image_byte_strings')
        image_encoder_inputs = preprocess_image_byte_strings(image_byte_strings)
        image_encoder_masks = tf.placeholder(tf.float32, shape=(batch_size, None),
                                             name='image_masks')
        image_encoder_params = config['model']['image_encoder_params']
    else:
        image_encoder_inputs = None
        image_encoder_masks = None
        image_encoder_params = {}

    # Tables.
    if config['model']['use_tables']:
        known_attrs = tf.placeholder(tf.int32, shape=(batch_size, None),
                                     name='known_attrs')
        known_values = tf.placeholder(tf.int32, shape=(batch_size, None),
                                      name='known_values')
        known_attr_embeddings = tf.nn.embedding_lookup(attr_embeddings,
                                                       known_attrs)
        known_value_embeddings = tf.nn.embedding_lookup(value_embeddings,
                                                        known_values)
        table_encoder_inputs = tf.concat([known_attr_embeddings,
                                          known_value_embeddings],
                                         axis=2)
        table_encoder_masks = tf.placeholder(tf.float32,
                                             shape=(batch_size, None),
                                             name='table_masks')
        table_encoder_params = config['model']['table_encoder_params']
    else:
        table_encoder_inputs = None
        table_encoder_masks = None
        table_encoder_params = {}

    # === Model Output and Loss Functions ===

    net_out, _ = mae(
        attr_queries,
        num_outputs=config['model']['context_embedding_size'],
        table_encoder_inputs=table_encoder_inputs,
        table_encoder_masks=table_encoder_masks,
        table_encoder_params=table_encoder_params,
        image_encoder_inputs=image_encoder_inputs,
        image_encoder_masks=image_encoder_masks,
        image_encoder_params=image_encoder_params,
        desc_encoder_inputs=desc_encoder_inputs,
        desc_encoder_masks=desc_encoder_masks,
        desc_encoder_params=desc_encoder_params,
        fusion_method=config['model']['fusion_method'])
    predicted_values = net_out

    # === Loss ===

    distance_metric=config['model']['distance_metric']
    s = utils.distance(predicted_values, correct_values, distance_metric)
    s_prime = utils.distance(predicted_values, incorrect_values,
                             distance_metric)
    loss = tf.maximum(0.0, 1.0 + s - s_prime)
    loss = tf.reduce_mean(loss)
    tf.losses.add_loss(loss)

    mean_loss, _ = tf.metrics.mean(loss,
                                   metrics_collections=['rank_metrics'],
                                   updates_collections=['rank_updates'],
                                   name='streaming_loss')
    tf.summary.scalar('loss', mean_loss)

    # === Evaluation ===

    # Identify <UNK> samples
    unk = tf.equal(correct_value_ids, config['data']['value_vocab_size'] - 1)
    obs = tf.logical_not(unk)

    # Compute scores
    scores = utils.distance_matrix(predicted_values, value_embeddings,
                                   distance_metric)
    scores = tf.identity(scores, name='scores')

    # Boolean matrix, True for elements where score is less than the score of
    # the correct value
    rank = utils.rank(scores, correct_value_ids)
    rank = tf.identity(rank, name='rank')

    # Mean reciprocal rank
    mrr, _ = tf.metrics.mean(1.0 / rank,
                             metrics_collections=['rank_metrics'],
                             updates_collections=['rank_updates'],
                             name='streaming_mrr')
    mrr_obs, _ = tf.metrics.mean(1.0 / tf.boolean_mask(rank, obs),
                                 metrics_collections=['rank_metrics'],
                                 updates_collections=['rank_updates'],
                                 name='streaming_mrr_obs')
    mrr_unk, _ = tf.metrics.mean(1.0 / tf.boolean_mask(rank, unk),
                                 metrics_collections=['rank_metrics'],
                                 updates_collections=['rank_updates'],
                                 name='streaming_mrr_unk')
    tf.summary.scalar('mean_reciprocal_rank', mrr)
    tf.summary.scalar('mean_reciprocal_rank_obs', mrr_obs)
    tf.summary.scalar('mean_reciprocal_rank_unk', mrr_unk)

    # Accuracy at k
    lt_1 = tf.cast(rank <= 1.0, dtype=tf.float32)
    lt_20 = tf.cast(rank <= 20.0, dtype=tf.float32)
    acc_at_1, _ = tf.metrics.mean(lt_1,
                                  metrics_collections=['rank_metrics'],
                                  updates_collections=['rank_updates'],
                                  name='streaming_acc_at_1')
    acc_at_1_obs, _ = tf.metrics.mean(tf.boolean_mask(lt_1, obs),
                                      metrics_collections=['rank_metrics'],
                                      updates_collections=['rank_updates'],
                                      name='streaming_acc_at_1_obs')
    acc_at_1_unk, _ = tf.metrics.mean(tf.boolean_mask(lt_1, unk),
                                      metrics_collections=['rank_metrics'],
                                      updates_collections=['rank_updates'],
                                      name='streaming_acc_at_1_unk')
    acc_at_20, _ = tf.metrics.mean(lt_20,
                                  metrics_collections=['rank_metrics'],
                                  updates_collections=['rank_updates'],
                                  name='streaming_acc_at_20')
    acc_at_20_obs, _ = tf.metrics.mean(tf.boolean_mask(lt_20, obs),
                                      metrics_collections=['rank_metrics'],
                                      updates_collections=['rank_updates'],
                                      name='streaming_acc_at_20_obs')
    acc_at_20_unk, _ = tf.metrics.mean(tf.boolean_mask(lt_20, unk),
                                      metrics_collections=['rank_metrics'],
                                      updates_collections=['rank_updates'],
                                      name='streaming_acc_at_20_unk')
    tf.summary.scalar('accuracy_at_1', acc_at_1)
    tf.summary.scalar('accuracy_at_1_obs', acc_at_1_obs)
    tf.summary.scalar('accuracy_at_1_unk', acc_at_1_unk)
    tf.summary.scalar('accuracy_at_20', acc_at_20)
    tf.summary.scalar('accuracy_at_20_obs', acc_at_20_obs)
    tf.summary.scalar('accuracy_at_20_unk', acc_at_20_unk)

    # Summarize attention weights
    # attn_vars = [i for i in tf.global_variables() if 'alpha' in i.name]
    # for attn_var in attn_vars:
    #     tf.summary.histogram(attn_var.name, attn_var)


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

    # If no checkpoint found, then check for VGG / embedding matrices
    vgg_ckpt = config['data']['vgg_ckpt']
    use_images = config['model']['use_images']
    if vgg_ckpt and use_images:
        vgg_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          scope="mae/image_encoder/vgg_16/conv")
        # Little name hackeroonie
        vgg_variables = {
            x.name.replace('mae/image_encoder/', '').replace(':0', ''): x for x in vgg_variables
        }
        vgg_saver = tf.train.Saver(vgg_variables)
        tf.logging.info('Using pretrained VGG weights from: %s' % vgg_ckpt)
    else:
        vgg_saver = None

    glove_ckpt = config['data']['glove_ckpt']
    use_descs = config['model']['use_descs']
    if glove_ckpt and use_descs:
        glove_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                            scope='desc_word_embeddings')
        glove_saver = tf.train.Saver(glove_variables)
        tf.logging.info('Using pretrained GloVe embeddings from: %s' %
                        glove_ckpt)
    else:
        glove_saver = None

    av_ckpt = config['data']['av_ckpt']
    if av_ckpt:
        av_variables = [
            *tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope='attr_embeddings'),
            *tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                               scope='value_embeddings')
        ]
        av_saver = tf.train.Saver(av_variables)
        tf.logging.info('Using pretrained attr/value embeddings from: %s' %
                        av_ckpt)
    else:
        av_saver = None

    def init_fn(sess):
        if vgg_saver:
            vgg_saver.restore(sess, vgg_ckpt)
        if glove_saver:
            glove_saver.restore(sess, glove_ckpt)
        if av_saver:
            av_saver.restore(sess, av_ckpt)

    return init_fn


def main(_):
    # Load config
    config = utils.load_config(FLAGS.config)

    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])
    ckpt = os.path.join(ckpt_dir, 'model.ckpt')

    log_dir = os.path.join(config['training']['log_dir'],
                           config['experiment_name'])
    if not os.path.exists(ckpt_dir):
        tf.logging.info('Creating checkpoint directory: %s' % ckpt_dir)
        os.mkdir(ckpt_dir)
    if not os.path.exists(log_dir):
        tf.logging.info('Creating log directory: %s' % log_dir)

    config_path = os.path.join(ckpt_dir, 'config.yaml')
    if os.path.exists(config_path):
        tf.logging.info('Existing configuration file detected.')
        if config != utils.load_config(config_path):
            if FLAGS.force:
                warnings.warn('Specified configuration does not match '
                              'existing configuration in checkpoint directory. '
                              'Forcing overwrite of existing configuration.')
                shutil.copyfile(FLAGS.config, config_path)
            else:
                raise ValueError('Specified configuration does not match '
                                 'existing configuration in checkpoint '
                                 'directory.')
    else:
        tf.logging.info('No existing configuration found. Copying config file '
                        'to "%s".' % config_path)
        shutil.copyfile(FLAGS.config, config_path)

    g = tf.Graph()
    with g.as_default():
        tf.logging.info('Creating graph')
        build_graph(config)
        global_step = tf.train.get_or_create_global_step()

        saver = tf.train.Saver(max_to_keep=5)

        init_fn = get_init_fn(config)

        total_loss = tf.losses.get_total_loss()
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_vars = [var for var in trainable_vars if 'vgg' not in var.name]
        learning_rate = tf.train.exponential_decay(
            learning_rate=config['training']['initial_learning_rate'],
            global_step=global_step,
            decay_steps=config['training']['decay_steps'],
            decay_rate=config['training']['decay_rate'],
            staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 clip_gradient_norm=config['training']['gradient_clipping'],
                                                 variables_to_train=trainable_vars,
                                                 summarize_gradients=True)
        summary_op = tf.summary.merge_all()
        eval_logger = tf.summary.FileWriter(log_dir)

        metric_op = tf.get_collection('rank_metrics')
        update_op = tf.get_collection('rank_updates')

        streaming_vars = [i for i in tf.local_variables() if 'streaming' in i.name]
        reset_op = [tf.variables_initializer(streaming_vars)]

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            init_fn(sess) # Responsible for restoring variables / warm starts

            # Generate data loop.
            for feed_dict, uris in utils.generate_batches('train', config):

                try:
                    i, _ = sess.run([global_step, train_op], feed_dict=feed_dict)
                except tf.errors.InvalidArgumentError: # Encountered a bad JPEG
                    continue

                if not i % config['training']['log_frequency']:
                    try:
                        loss = sess.run(total_loss, feed_dict=feed_dict)
                    except tf.errors.InvalidArgumentError: # Encountered a bad JPEG
                        continue
                    tf.logging.info('Iteration %i - Loss: %0.4f' % (i, loss))

                if not i % config['training']['save_frequency']:
                    tf.logging.info('Saving checkpoint for iteration %i' % i)
                    saver.save(sess, ckpt)

                    sess.run(reset_op)
                    # Evaluate on test data.
                    for feed_dict, uris in utils.generate_batches('val', config):
                        try:
                            sess.run(update_op, feed_dict=feed_dict)
                        except tf.errors.InvalidArgumentError:
                            continue
                    print(sess.run(metric_op))

                    # Write summaries.
                    summary = sess.run(summary_op, feed_dict=feed_dict)
                    eval_logger.add_summary(summary, i)

                if i >= config['training']['max_steps']:
                    tf.logging.info('Training complete')
                    sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='The configuration file.')
    parser.add_argument('-f', '--force', action='store_true',
                        help='Forces model to train even if configuration '
                        'disagrees with existing configuration. Existing '
                        'configuration will be overwritten.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

