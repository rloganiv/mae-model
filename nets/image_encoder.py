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
"""Contains the model definition for the image encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range

import tensorflow as tf

from nets.vgg import vgg_16, vgg_arg_scope

slim = tf.contrib.slim


def visual_attention(inputs,
                     contexts,
                     reuse=None,
                     scope=None):
    """Performs visual attention on individual images."""
    batch_size = inputs.shape[0]
    attr_embedding_size = contexts.shape[-1]
    with tf.variable_scope(scope, 'visual_attention',
                           [inputs, contexts],
                           reuse=reuse) as sc:
        # Copy attr queries to make concatenation possible.
        contexts = tf.reshape(contexts,
                                  [batch_size, 1, 1, attr_embedding_size])
        contexts = tf.tile(contexts,
                               [1, tf.shape(inputs)[1], 196, 1])
        # See pg. 4 of 'Show, Attend and Tell' paper.
        z = tf.concat([inputs, contexts], axis=3)
        f_att = slim.fully_connected(z, num_outputs=1)
        alpha = tf.nn.softmax(f_att, name='alpha', dim=2)
        phi = tf.reduce_sum(alpha * inputs, axis=2)
        return phi, alpha


def image_seq_attention(inputs,
                        contexts,
                        masks,
                        reuse=None,
                        scope=None):
    """Performs attention to aggregate sequences of images."""
    batch_size = inputs.shape[0]
    attr_embedding_size = contexts.shape[-1]
    with tf.variable_scope(scope, 'image_seq_attention',
                           [inputs, contexts],
                           reuse=reuse) as sc:
        # Copy contexts to make concatenation possible.
        contexts = tf.reshape(contexts, [batch_size, 1, attr_embedding_size])
        contexts = tf.tile(contexts, [1, tf.shape(inputs)[1], 1])

        z = tf.concat([inputs, contexts], axis=2)
        f_att = slim.fully_connected(z, num_outputs=1)
        alpha = tf.nn.softmax(f_att, dim=1)
        # Renormalize after applying masks
        alpha = tf.multiply(alpha, masks)
        alpha = tf.div(alpha, tf.reduce_sum(alpha, axis=1, keep_dims=True),
                       name='alpha')
        phi = tf.reduce_sum(alpha * inputs, axis=1)
        return phi, alpha


def image_encoder(inputs,
                  masks,
                  num_outputs=1,
                  contexts=None,
                  is_training=False,
                  dropout_keep_prob=0.5,
                  use_attention=False,
                  reuse=None,
                  scope=None,
                  **kwargs):
    """Image encoder network.

    Args:
        inputs: A tensor of size [batch_size, n_images, height, width, channels].
        masks: A tensor of size [batch_size, n_images] used to mask padded
            values in the input tensor.
        num_ouputs: Size of network output.
        is_training: Whether or not the network is being trained.
        dropout_keep_prob: Probability that activations are kept in dropout
            layers during training.
        use_attention: Whether or not to use attention to aggregate model
            outputs. If False then aggregation is done using max_pooling
            instead. If True then contexts must be specified.
        contexts: A tensor of size [batch_size, embedding_size] containing
            contextsual information to be used to compute attention.
            NOTE: Not used if `use_attention` is False.
        reuse: Whether or not the network and its variables should be reused.
            Requires that 'scope' is given.
        scope: Optional scope for the variables.

    Returns:
        net: The output of the network.
        end_points: A dict of tensors with intermediate activations.
    """
    batch_size = inputs.shape[0]
    if use_attention:
        assert contexts is not None, 'Missing attr queries.'
    with tf.variable_scope(scope, 'image_encoder', [inputs, masks],
                           reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Combine batch and sequence length dims so that tensor shape
        # matches expected input shape for vgg_16
        net = tf.reshape(inputs, [-1, 224, 224, 3])
        with slim.arg_scope(vgg_arg_scope(1e-5)):
            _, vgg_end_points = vgg_16(net, is_training=is_training,
                                   dropout_keep_prob=dropout_keep_prob)
            if use_attention:
                net = vgg_end_points[sc.name + '/vgg_16/conv5/conv5_3']
                net = tf.reshape(net, [batch_size, -1, 196, 512])
                net, alpha = visual_attention(net, contexts)
                tf.add_to_collection(end_points_collection, alpha)
            else:
                net = vgg_end_points[sc.name + '/vgg_16/fc7']
                net = tf.reshape(net, [batch_size, -1, 4096])

        # Get representation
        net = slim.dropout(net, dropout_keep_prob, scope='dropout')
        net = slim.fully_connected(net, num_outputs=num_outputs, scope='fc')

        # Combine image-wise representations into single representation
        masks = tf.expand_dims(masks, 2) # To make broadcasting work
        if use_attention:
            net, alpha = image_seq_attention(net, contexts, masks)
            tf.add_to_collection(end_points_collection, alpha)
        else:
            # Max pool over time
            net = tf.multiply(net, masks)
            net = tf.reduce_max(net, axis=1, name='one_max_pooling')
            # Late fusion
            net = tf.concat([net, contexts], axis=1)
            net = slim.dropout(net, dropout_keep_prob,
                               scope='dropout_late_fusion')
            net = slim.fully_connected(net, num_outputs=num_outputs,
                                       scope='fc_late_fusion')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        end_points.update(vgg_end_points)
        return net, end_points

