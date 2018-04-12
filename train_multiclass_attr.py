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
"""Multiclass attribute extractor model training script.

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

from nets.multiclass_attr import multiclass_attr
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
    attr_ids = tf.placeholder(tf.int32,
                              shape=(batch_size,None),
                              name='attr_ids')
    attrs = tf.one_hot(attr_ids,
                       depth=config['data']['attr_vocab_size'])
    attrs = tf.reduce_sum(attrs, axis=1)

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


    # === Model Output and Loss Functions ===

    logits, _ = multiclass_attr(
        num_outputs=config['data']['attr_vocab_size'],
        image_encoder_inputs=image_encoder_inputs,
        image_encoder_masks=image_encoder_masks,
        image_encoder_params=image_encoder_params,
        desc_encoder_inputs=desc_encoder_inputs,
        desc_encoder_masks=desc_encoder_masks,
        desc_encoder_params=desc_encoder_params)

    # === Loss ===

    loss = tf.losses.sigmoid_cross_entropy(attrs, logits)
    mean_loss, _ = tf.metrics.mean(loss,
                                   metrics_collections=['metrics'],
                                   updates_collections=['updates'],
                                   name='streaming_loss')
    tf.summary.scalar('loss', mean_loss)

    # === Evaluation ===

    predictions = tf.round(tf.sigmoid(logits))
    accuracy, _ = tf.metrics.accuracy(attrs,
                                      predictions,
                                      metrics_collections=['metrics'],
                                      updates_collections=['updates'],
                                      name='streaming_accuracy')
    precision, _ = tf.metrics.precision(attrs,
                                        predictions,
                                        metrics_collections=['metrics'],
                                        updates_collections=['updates'],
                                        name='streaming_precision')
    recall, _ = tf.metrics.recall(attrs,
                                  predictions,
                                  metrics_collections=['metrics'],
                                  updates_collections=['updates'],
                                  name='streaming_recall')
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('precision', precision)
    tf.summary.scalar('recall', recall)


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

    def init_fn(sess):
        if vgg_saver:
            vgg_saver.restore(sess, vgg_ckpt)
        if glove_saver:
            glove_saver.restore(sess, glove_ckpt)
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

        metric_op = tf.get_collection('metrics')
        update_op = tf.get_collection('updates')

        streaming_vars = [i for i in tf.local_variables() if 'streaming' in i.name]
        reset_op = [tf.variables_initializer(streaming_vars)]

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            init_fn(sess) # Responsible for restoring variables / warm starts

            # Generate data loop.
            for feed_dict, uris in utils.generate_batches('train', config,
                                                          mc_attr=True):

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
                    for feed_dict, uris in utils.generate_batches('val', config,
                                                                  mc_attr=True):
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

