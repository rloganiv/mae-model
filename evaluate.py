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

    # Setup output writer
    output_file = open(FLAGS.output, 'w', newline='')
    output_writer = csv.writer(output_file, delimiter=',')

    # Get checkpoint dir
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])
    ckpt = os.path.join(ckpt_dir, 'model.ckpt')


    # Build graph
    g = tf.Graph()
    with g.as_default():
        tf.logging.info('Creating graph')
        build_graph(config)
        saver = tf.train.Saver()
        metric_op = tf.get_collection('rank_metrics')
        update_op = tf.get_collection('rank_updates')
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            sess.run([tf.local_variables_initializer()])

            for i, batch in enumerate(utils.generate_batches('test', config)):
                try:
                    attr_query_id = batch['attr_query_ids:0'][0]
                    correct_value_id = batch['correct_value_ids:0'][0]
                    rank, scores = sess.run(['rank:0', 'scores:0'], feed_dict=batch)
                    output_writer.writerow([attr_query_id, correct_value_id,
                                            rank[0], *scores[0]])

                except tf.errors.InvalidArgumentError: # A bad JPEG
                    continue

    # Close writer
    output_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='The configuration file.')
    parser.add_argument('output', type=str, help='Where to output results.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

