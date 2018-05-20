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
"""MAE model evaluation script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import numpy as np
import os
import tensorflow as tf

import utils
from train import build_graph

slim = tf.contrib.slim


FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    # Get config
    config = utils.load_config(FLAGS.config)
    config['training']['batch_size'] = 1
    config['training']['max_desc_length'] = 3000
    config['training']['max_number_of_images'] = 3000
    config['model']['image_encoder_params']['dropout_keep_prob']=1.00

    # Setup output writer
    output_file = open(FLAGS.output, 'w', newline='')
    output_writer = csv.writer(output_file, delimiter=',')

    # Get checkpoint dir
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])
    ckpt = os.path.join(ckpt_dir, 'model.ckpt-best')

    # Build graph
    g = tf.Graph()
    with g.as_default():
        tf.logging.info('Creating graph')
        build_graph(config, is_training=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            for feed_dict, uris in utils.generate_batches('test', config,
                                                          fnames=FLAGS.fnames):

                attr_query_id = feed_dict['attr_query_ids:0'][0]
                correct_value_id = feed_dict['value_ids:0'][0]
                try:
                    logits = sess.run('mae/fully_connected/BiasAdd:0', feed_dict=feed_dict)
                except tf.errors.InvalidArgumentError:
                    continue
                predicted_value_id = np.argmax(logits)
                output_writer.writerow([uris[0], attr_query_id,
                                        correct_value_id, predicted_value_id,
                                        *logits[0]])

    # Close writer
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='The configuration file.')
    parser.add_argument('output', type=str, help='Where to output results.')
    parser.add_argument('fnames', type=str, nargs='+')
    FLAGS, _ = parser.parse_known_args()

    main(_)

