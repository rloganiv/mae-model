"""MAE model training script.

Usage:
    python train.py --config config.yaml
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tensorflow as tf

from nets.mae import mae
import utils

slim = tf.contrib.slim


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('config', '',
                       'Configuration file.')

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


def cosine_similarity(x1, x2, pairwise=False):
    x1 = tf.nn.l2_normalize(x1, dim=0)
    x2 = tf.nn.l2_normalize(x2, dim=0)
    if pairwise:
        return tf.tensordot(x1, x2, axes=[[1], [1]])
    else:
        return tf.reduce_sum(x1 * x2, axis=1)


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
                   config['model']['embedding_size']))
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

    predicted_values, _ = mae(
        attr_queries,
        num_outputs=config['model']['embedding_size'],
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

    s = cosine_similarity(predicted_values, correct_values)
    s_prime = cosine_similarity(predicted_values, incorrect_values)
    loss = tf.maximum(0.0, s_prime - s + 1) # Question: Is there a better margin?
    loss = tf.reduce_mean(loss)
    tf.losses.add_loss(loss)
    tf.summary.scalar('loss', loss)

    # === Evaluation ===

    # Compute scores
    s = cosine_similarity(predicted_values, value_embeddings, pairwise=True)

    # Boolean matrix, True for elements where score is less than the score of
    # the correct value
    incorrect = s >= tf.gather(s, correct_value_ids)
    incorrect = tf.cast(incorrect, dtype=tf.float32)
    mrr = tf.reduce_mean(1.0 / tf.reduce_sum(incorrect, axis=1))
    tf.summary.scalar('mean_reciprocal_rank', mrr)


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
    vgg_ckpt = config['training']['vgg_ckpt']
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

    # TODO: Embedding matrices
    embedding_saver = None

    def init_fn(sess):
        if vgg_saver:
            vgg_saver.restore(sess, vgg_ckpt)
        if embedding_saver:
            pass

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
            raise ValueError('Specified configuration does not match '
                             'existing configuration in checkpoint directory.')
    else:
        tf.logging.info('Copying config to ckpt directory.')
        shutil.copyfile(FLAGS.config, config_path)

    g = tf.Graph()
    with g.as_default():
        tf.logging.info('Creating graph')
        build_graph(config)
        saver = tf.train.Saver(max_to_keep=5)

        init_fn = get_init_fn(config)

        total_loss = tf.losses.get_total_loss()
        optimizer = tf.train.AdamOptimizer(config['training']['learning_rate'])
        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)
        summary_op = tf.summary.merge_all()
        eval_logger = tf.summary.FileWriter(log_dir)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            init_fn(sess) # Responsible for restoring variables / warm starts

            # Generate data loop.
            for i, batch in enumerate(utils.generate_batches('train', config)):
                # Run train_op on feed_dict.
                try:
                    sess.run(train_op, feed_dict=batch)
                except tf.errors.InvalidArgumentError: # Encountered a bad JPEG
                    continue
                if not i % config['training']['log_frequency']:
                    loss = sess.run(total_loss, feed_dict=batch)
                    tf.logging.info('Iteration %i - Loss: %0.4f' % (i, loss))

                if not i % config['training']['save_frequency']:
                    tf.logging.info('Saving checkpoint for iteration %i' % i)
                    saver.save(sess, ckpt)
                    summary = sess.run(summary_op, feed_dict=batch)
                    eval_logger.add_summary(summary, i)


if __name__ == '__main__':
    tf.app.run()

