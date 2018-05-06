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
"""Contains the model definition for the GRU description encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

slim = tf.contrib.slim


# TODO: Maybe use a different activation fn than `tanh`.
def _gru_cell(size,
              dropout_keep_prob):
    cell = tf.contrib.rnn.GRUCell(size)
    wrapped_cell = tf.contrib.rnn.DropoutWrapper(
        cell,
        input_keep_prob=1.0,
        output_keep_prob=dropout_keep_prob,
        state_keep_prob=dropout_keep_prob)
    return wrapped_cell

def gru_desc_encoder(inputs,
                     mask,
                     num_outputs,
                     hidden_size=512,
                     num_layers=2,
                     contexts=None,
                     is_training=False,
                     dropout_keep_prob=0.5,
                     reuse=None,
                     scope=None,
                     **kwargs):
    """GRU text encoder."""
    assert hidden_size % 2 == 0, \
            'Number of hidden units must be even.'
    with tf.variable_scope(scope, 'text_encoder', [inputs],
                           reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Need to obtain lengths of sequences from mask for dynamic_rnn.
        with tf.variable_scope('bidirectional_rnn'):
            sequence_length = tf.reduce_sum(mask, axis=1)
            sequence_length = tf.cast(sequence_length, dtype=tf.int32)
            if is_training:
                cells_fw = [_gru_cell(hidden_size / 2, dropout_keep_prob)]
                cells_bw = [_gru_cell(hidden_size / 2, dropout_keep_prob)]
            else:
                cells_fw = [_gru_cell(hidden_size / 2, 1.0)]
                cells_bw = [_gru_cell(hidden_size / 2, 1.0)]
            _, out_fw, out_bw = stack_bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                inputs,
                sequence_length=sequence_length,
                dtype=tf.float32)
            # Note: -1 index to make sure we only use the outputs of the last
            # layer.
            net = tf.concat([out_fw[-1], out_bw[-1]], axis=-1)
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        # FC layers
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, num_outputs, scope='fc1')
        end_points[sc.name + '/fc1'] = net

        # Late fusion of contexts
        if contexts is not None:
            net = tf.concat([net, contexts], axis=1)

        # FC2
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, num_outputs, scope='fc2')
        end_points[sc.name + '/fc2'] = net
        return net, end_points

