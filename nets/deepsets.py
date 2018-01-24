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
"""Contains the model definition for the Deep Sets network.

The model definition was introduced in following conference paper:

    Deep Sets
    Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos,
    Ruslan R Salakhutdinov, and Alexander J Smola
    Advances in Neural Information Processing Systems 30
    PDF: https://papers.nips.cc/paper/6931-deep-sets.pdf

Usage:
    with slim.arg_scope(deepsets.deepsets_arg_scope()):
        net, end_points = deepsets.deepsets(inputs, masks, contexts)

@@deepsets_v0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

import tensorflow as tf

slim = tf.contrib.slim


def deepsets_arg_scope(l2_reg_scale=0.0005):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(l2_reg_scale),
                        biases_initializer=tf.zeros_initializer()) as arg_sc:
        return arg_sc


def deepsets(inputs,
             masks,
             num_outputs=1,
             contexts=None,
             is_training=False,
             dropout_keep_prob=0.5,
             phi_layers=3,
             phi_units=128,
             rho_layers=3,
             rho_units=128,
             reuse=None,
             scope=None,
             **kwargs):
    """Permutation invariant Deep Sets architecture.

    Args:
        inputs: A tensor of size [batch_size, max_set_size, embedding_size]
            representing the set elements.
        masks: A tensor of size [batch_size, max_set_size] to mask padded
            values in the inputs tensor.
        contexts: Optional tensor of size [batch_size, contexts_embedding_size]
            containing contextsual information to be concatenated with the set
            feature obtained from pooling individual element representation.
        num_outputs: Size of network output.
        is_training: Whether or not the network is being trained.
        dropout_keep_prob: Probability that activations are kept in dropout
            layers during training.
        phi_layers: Number of layers in the elementwise network.
        phi_units: Number of units in the elementwise network.
        rho_layers: Number of layers in the setwise network.
        rho_units: Number of units in the setwise network.
        reuse: Whether or not the network and its variables should be reused.
            Requires that 'scope' is given.
        scope: Optional scope for the variables.

    Returns:
        net: The output of the network.
        end_points: A dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'deepsets', [inputs, masks, contexts],
                           reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs of fully connected layers
        with slim.arg_scope([slim.fully_connected],
                            outputs_collections=end_points_collection):
            net = inputs
            # Elementwise transformation (phi) network.
            with tf.variable_scope('phi'):
                for i in range(phi_layers):
                    net = slim.fully_connected(net, phi_units,
                                               scope='fc_%i' % (i+1))
                    net = slim.dropout(net, dropout_keep_prob,
                                       is_training=is_training,
                                       scope='dropout_%i' % (i+1))
            # Mask and pool elementwise features.
            masks = tf.expand_dims(masks, 2)
            net = tf.multiply(net, masks, name='mask')
            net = tf.reduce_sum(net, axis=1, name='sum_pool')
            # Add contextsual information if provided.
            if contexts is not None:
                net = tf.concat([net, contexts], axis=1, name='concat_contexts')
            # Setwise transformation (rho) network.
            with tf.variable_scope('rho'):
                for i in range(rho_layers - 1):
                    net = slim.fully_connected(net, rho_units,
                                               scope='fc_%i' % (i+1))
                    net = slim.dropout(net, dropout_keep_prob,
                                         is_training=is_training,
                                         scope='rho_dropout_%i' % (i+1))
                net = slim.fully_connected(net, num_outputs, activation_fn=None,
                                           scope='fc_%i' % (rho_layers))
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points

