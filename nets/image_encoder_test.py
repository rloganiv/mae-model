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
"""Tests for image encoder architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from image_encoder import image_encoder


class ImageEncoderTest(tf.test.TestCase):

    def testBuild(self):
        batch_size = 16
        n_images = 4
        height = 224
        width = 224
        channels = 3
        attr_embedding_size = 12
        num_outputs=128
        with self.test_session():
            inputs = tf.random_uniform([batch_size, n_images, height, width,
                                        channels])
            masks = tf.ones([batch_size, n_images])
            contexts = tf.random_uniform([batch_size, attr_embedding_size])
            outputs, _ = image_encoder(inputs, masks, num_outputs=num_outputs,
                                       use_attention=False, contexts=contexts)
            self.assertListEqual(outputs.get_shape().as_list(),
                                 [batch_size, num_outputs])

    def testBuildWithAttention(self):
        batch_size = 16
        n_images = 4
        height = 224
        width = 224
        channels = 3
        attr_embedding_size = 12
        num_outputs=128
        with self.test_session():
            inputs = tf.random_uniform([batch_size, n_images, height, width,
                                        channels])
            masks = tf.ones([batch_size, n_images])
            contexts = tf.random_uniform([batch_size, attr_embedding_size])
            outputs, _ = image_encoder(inputs, masks, num_outputs=num_outputs,
                                       use_attention=True, contexts=contexts)
            self.assertListEqual(outputs.get_shape().as_list(),
                                 [batch_size, num_outputs])


if __name__ == '__main__':
    tf.test.main()

