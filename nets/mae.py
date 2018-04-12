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
"""Contains the model definition for the Multimodal Attribute Extraction model.

The model definition was introduces in the following paper:

    To be accepted by a major conference/journal

Usage:
    with slim.arg_scope(mae.mae_arg_scope()):
        net, end_points = mae.mae(DEFINE INPUTS)

@@mae
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers

from nets.deepsets import deepsets, deepsets_arg_scope
from nets.desc_encoder import desc_encoder
from nets.image_encoder import image_encoder

slim = tf.contrib.slim


def encoding_attention(inputs,
                       contexts,
                       reuse=None,
                       scope=None):
    """Performs attention to aggregate encodings."""
    batch_size = inputs[0].shape[0]
    attr_embedding_size = contexts.shape[-1]
    with tf.variable_scope(scope, 'encoding_attention',
                           [inputs, contexts],
                           reuse=reuse) as sc:
        # Copy contexts to make concatenation possible.
        net = tf.stack(inputs, axis=1)
        contexts = tf.reshape(contexts, [batch_size, 1, attr_embedding_size])
        contexts = tf.tile(contexts, [1, len(inputs), 1])

        z = tf.concat([net, contexts], axis=2)
        f_att = slim.fully_connected(z, num_outputs=1)
        alpha = tf.nn.softmax(f_att, dim=1, name='alpha')
        phi = tf.reduce_sum(alpha * net, axis=1)
        return phi, alpha


def mae(attr_queries,
        num_outputs=1,
        table_encoder_inputs=None,
        table_encoder_masks=None,
        table_encoder_params={},
        image_encoder_inputs=None,
        image_encoder_masks=None,
        image_encoder_params={},
        desc_encoder_inputs=None,
        desc_encoder_masks=None,
        desc_encoder_params={},
        is_training=False,
        fusion_method='concat',
        reuse=None,
        scope=None):
    """Multimodal attribute extraction architecture.

    Args:
        attr_queries: Tensor of size [batch_size, embedding_size].
        num_outputs: Size of network output.
        image_encoder_inputs: Tensor of size [batch_size, n_images, 224, 244, 3] to be
            fed into image encoder architecture.
        image_encoder_masks: Tensor of size [batch_size, n_images] to mask padded
            values in image_inputs.
        image_encoder_params: Dict of parameters to be used in image encoder.
        table_encoder_inputs: Tensor of size [] to be fed into Deep Sets
            architecture.
        table_encoder_masks: Tensor of size [] to mask padded values in
            table_inputs.
        table_encoder_params: Dict of parameters to be used in Deep Sets
            archtitecture.
        text_encoder_inputs: Tensor of size [] to be fed into text encoder
            architecture.
        text_encoder_params: Dict of parameters to be used in text encoder.
        fusion_method: One of 'concat' or 'attention'.
        reuse: Whether or not the network and its variables should be reused.
            Requires that 'scope' is given.
        scope: Optional scope for the variables.

    Returns:
        net: The output of the network.
        end_points: A dict of tensors with intermediate activations.
    """
    model_inputs = [
        attr_queries,
        desc_encoder_inputs,
        image_encoder_inputs,
        image_encoder_masks,
        table_encoder_inputs,
        table_encoder_masks,
    ]

    with tf.variable_scope(scope, 'mae', model_inputs, reuse=reuse) as sc:
        branches = [attr_queries]
        end_points_collection = sc.original_name_scope + '_end_points'

        if desc_encoder_inputs is not None:
            if desc_encoder_masks is None:
                raise ValueError('desc_encoder_masks must be specified.')
            desc_encoding, desc_end_points = desc_encoder(
                desc_encoder_inputs,
                desc_encoder_masks,
                contexts=attr_queries,
                is_training=is_training,
                **desc_encoder_params)
            branches.append(desc_encoding)

        if image_encoder_inputs is not None:
            if image_encoder_masks is None:
                raise ValueError('image_encoder_masks must be specified.')
            image_encoding, image_end_points = image_encoder(
                image_encoder_inputs,
                image_encoder_masks,
                contexts=attr_queries,
                is_training=is_training,
                **image_encoder_params)
            branches.append(image_encoding)

        if table_encoder_inputs is not None:
            if table_encoder_masks is None:
                raise ValueError('table_encoder_masks must be specified.')
            with slim.arg_scope(deepsets_arg_scope()):
                table_encoding, table_end_points = deepsets(
                    table_encoder_inputs,
                    table_encoder_masks,
                    contexts=attr_queries,
                    is_training=is_training,
                    **table_encoder_params)
            branches.append(table_encoding)

        if len(branches) == 1:
            net = branches[0]
        elif fusion_method=='concat':
            net = tf.concat(branches, axis=1)
            net = slim.fully_connected(net, num_outputs=num_outputs)
        elif fusion_method=='attention':
            net, alpha = encoding_attention(branches, contexts=attr_queries)
            tf.add_to_collection(end_points_collection, alpha)

        # Additional FC layer.
        net = slim.fully_connected(net, num_outputs=num_outputs)

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        if desc_encoder_inputs is not None:
            end_points.update(desc_end_points)
        if image_encoder_inputs is not None:
            end_points.update(image_end_points)
        if table_encoder_inputs is not None:
            end_points.update(table_end_points)

        return net, end_points

