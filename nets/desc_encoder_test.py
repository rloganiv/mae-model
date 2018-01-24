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
"""Tests for text encoder architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from desc_encoder import desc_encoder


class DescEncoderTest(tf.test.TestCase):

    def testBuild(self):
        batch_size = 5
        seq_length = 50
        word_embedding_size = 16
        context_size = 64
        num_outputs = 124
        with self.test_session():
            inputs = tf.random_uniform([batch_size, seq_length,
                                        word_embedding_size])
            masks = tf.ones([batch_size, seq_length])
            contexts = tf.random_uniform([batch_size, context_size])
            outputs, _ = desc_encoder(inputs, masks, num_outputs, contexts)
            self.assertListEqual(outputs.get_shape().as_list(),
                                 [batch_size, num_outputs])


if __name__ == '__main__':
    tf.test.main()

