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
"""MATE model evaluation script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import numpy as np
import os
import tensorflow as tf

import utils
from train_mate import build_graph

slim = tf.contrib.slim


FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def greedy_table_search(sess, batch, config):
    batch['known_attrs:0'] = [[]]
    results = []
    unpicked_attr_ids = list(range(config['data']['attr_vocab_size']))
    for _ in range(len(unpicked_attr_ids)):
        scores = []
        # Get scores for attributes yet to be picked.
        for attr_id in unpicked_attr_ids:
            batch['pos_attr_query_ids:0'] = [attr_id]
            score = sess.run('scores:0', feed_dict=batch)[0][0]
            scores.append(float(score))
        # Find best score
        best_score_id = np.argmax(scores)
        best_score = scores[best_score_id]
        best_attr_id = unpicked_attr_ids[best_score_id]
        # Add to known attrs and results
        batch['known_attrs:0'][0].append(best_attr_id)
        results.append((best_attr_id, best_score))
        # Remove from unpicked
        unpicked_attr_ids = [id for id in unpicked_attr_ids if id != best_attr_id]
    return results


def main(_):
    # Get config
    config = utils.load_config(FLAGS.config)
    config['training']['batch_size'] = 1


    # Get checkpoint dir
    ckpt_dir = os.path.join(config['training']['ckpt_dir'],
                            config['experiment_name'])
    ckpt = os.path.join(ckpt_dir, 'model.ckpt')

    # Build graph
    results = []
    g = tf.Graph()
    with g.as_default():
        tf.logging.info('Creating graph')
        build_graph(config)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            sess.run([tf.local_variables_initializer()])

            for i, batch in enumerate(utils.generate_batches('test', config, mate=True)):
                batch, product = batch
                try:
                    results.append((greedy_table_search(sess, batch, config), product))
                except tf.errors.InvalidArgumentError: # A bad JPEG
                    continue
                if not (i+1) % 10000:
                    # Setup output writer
                    with open(FLAGS.output, 'w') as f:
                        json.dump(results, f)
                    break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='The configuration file.')
    parser.add_argument('output', type=str, help='Where to output results.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

