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
"""Tests for Deep Sets architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from deepsets import deepsets


class DeepSetsTest(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5
        max_set_size = 10
        embedding_size = 2
        with self.test_session():
            inputs = tf.random_uniform([batch_size, max_set_size,
                                        embedding_size])
            masks = tf.ones([batch_size, max_set_size])
            scores, _ = deepsets(inputs, masks)
            self.assertEqual(scores.op.name, 'deepsets/rho/fc_3/BiasAdd')
            self.assertListEqual(scores.get_shape().as_list(), [batch_size, 1])

    def testContexts(self):
        batch_size = 5
        max_set_size = 10
        embedding_size = 2
        context_size = 2
        num_outputs = 124
        with self.test_session():
            inputs = tf.random_uniform([batch_size, max_set_size,
                                        embedding_size])
            masks = tf.ones([batch_size, max_set_size])
            contexts = tf.random_uniform([batch_size, context_size])
            outputs, _ = deepsets(inputs, masks, num_outputs, contexts)
            self.assertListEqual(outputs.get_shape().as_list(),
                                 [batch_size, num_outputs])

    def testEndPoints(self):
        batch_size = 5
        max_set_size = 10
        embedding_size = 2
        with self.test_session():
            inputs = tf.random_uniform([batch_size, max_set_size,
                                        embedding_size])
            masks = tf.ones([batch_size, max_set_size])
            _, end_points = deepsets(inputs, masks)
            expected_names = ['deepsets/phi/fc_1',
                              'deepsets/phi/fc_2',
                              'deepsets/phi/fc_3',
                              'deepsets/rho/fc_1',
                              'deepsets/rho/fc_2',
                              'deepsets/rho/fc_3']
            self.assertSetEqual(set(end_points.keys()), set(expected_names))


if __name__ == '__main__':
    tf.test.main()

