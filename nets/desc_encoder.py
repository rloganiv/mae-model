# Copyright 2018 The MAE Authors. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 (the "License");
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
"""Contains the model definition for the description encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

import tensorflow as tf

slim = tf.contrib.slim


def desc_encoder(inputs,
                 masks,
                 num_outputs=1,
                 contexts=None,
                 is_training=False,
                 dropout_keep_prob=0.5,
                 window_sizes=[7],
                 num_filters=600,
                 reuse=None,
                 scope=None,
                 **kwargs):

    """Text encoder network.

    Args:
        inputs: A tensor of size [batch_size, seq_length, word_embedding_size].
        masks: A tensor of size [batch_size, seq_length] used to mask padded
            values in the input tensor.
        num_outputs: Size of network output.
        is_training: Whether or not the network is being trained.
        dropout_keep_prob: Probability that activations are kept in dropout
            layers during training.
        window_sizes: How many words are considered at a time for a given
            window.
        num_filters: Total number of filters to use.
        reuse: Whether or not the network and its variables should be reused.
            Requires that 'scope' is given.
        scope: Optional scope for the variables.

    Returns:
        net: The output of the network.
        end_points: A dict of tensors with intermediate activations.
    """
    assert num_filters % len(window_sizes) == 0, \
        'Number of windows must evenly divide number of filters.'
    filters_per_window = num_filters // len(window_sizes)

    with tf.variable_scope(scope, 'text_encoder', [inputs],
                           reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d], padding='VALID',
                            outputs_collections=end_points_collection):
            net = tf.expand_dims(inputs, 3) # Add fake 'channels' dimension
            branches = []
            for window_size in window_sizes:
                with tf.variable_scope('window_size_%i' % window_size):
                    branch = slim.conv2d(net, filters_per_window, 
                                         [window_size, inputs.shape[-1]])
                    branch = tf.squeeze(branch, axis=2)
                    truncated_masks = tf.expand_dims(masks[:, :-window_size+1], 2)
                    branch = branch * truncated_masks
                    branch = tf.reduce_max(branch, axis=1, name='one_max_pooling')
                    branches.append(branch)
        if contexts is not None:
            branches.append(contexts)
        net = tf.concat(branches, axis=1)
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, num_outputs, scope='fc')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        end_points[sc.name + '/fc'] = net
        return net, end_points

