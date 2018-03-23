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
"""Contains the model definition for the Multimodal Attribute Table Extraction
model.

The model definition was introduces in the following paper:

    To be accepted by a major conference/journal

Usage:
    net, end_points = mate.mate(DEFINE INPUTS)

@@mate
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


def mate(attr_queries,
         known_attrs,
         table_masks,
         deepsets_params={},
         image_encoder_inputs=None,
         image_encoder_masks=None,
         image_encoder_params={},
         desc_encoder_inputs=None,
         desc_encoder_masks=None,
         desc_encoder_params={},
         is_training=False,
         reuse=None,
         scope=None):
    """Multimodal attribute table extraction architecture.

    Args:
        attr_queries: Tensor of size [batch_size, embedding_size].
        known_attrs: Tensor of size [batch_size, table_size, embedding_size].
        table_masks: Tensor of size [batch_size, table_size].
        deepsets_params: Dict of parameters to be used in the Deep Sets model.
        image_encoder_inputs: Tensor of size [batch_size, n_images, 224, 244, 3] to be
            fed into image encoder architecture.
        image_encoder_masks: Tensor of size [batch_size, n_images] to mask padded
            values in image_inputs.
        image_encoder_params: Dict of parameters to be used in image encoder.
        text_encoder_inputs: Tensor of size [] to be fed into text encoder
            architecture.
        text_encoder_params: Dict of parameters to be used in text encoder.
        reuse: Whether or not the network and its variables should be reused.
            Requires that 'scope' is given.
        scope: Optional scope for the variables.

    Returns:
        net: The output of the network.
        end_points: A dict of tensors with intermediate activations.
    """
    model_inputs = [
        attr_queries,
        known_attrs,
        table_masks,
        image_encoder_inputs,
        image_encoder_masks,
        desc_encoder_inputs,
        desc_encoder_masks
    ]
    batch_size = known_attrs.shape[0]
    with tf.variable_scope(scope, 'mate', model_inputs, reuse=reuse) as sc:
        # Embed contexts
        context_branches = []
        end_points_collection = sc.original_name_scope + '_end_points'

        if desc_encoder_inputs is not None:
            if desc_encoder_masks is None:
                raise ValueError('desc_encoder_masks must be specified.')
            desc_encoding, desc_end_points = desc_encoder(
                desc_encoder_inputs,
                desc_encoder_masks,
                is_training=is_training,
                **desc_encoder_params)
            context_branches.append(desc_encoding)

        if image_encoder_inputs is not None:
            if image_encoder_masks is None:
                raise ValueError('image_encoder_masks must be specified.')
            image_encoding, image_end_points = image_encoder(
                image_encoder_inputs,
                image_encoder_masks,
                is_training=is_training,
                **image_encoder_params)
            context_branches.append(image_encoding)

        contexts = tf.concat(context_branches, axis=1)

        # Embed source data

        attr_queries = tf.expand_dims(attr_queries, 1)
        inputs = tf.concat([attr_queries, known_attrs], axis=1)

        tmp = tf.ones((batch_size, 1))
        masks = tf.concat([tmp, table_masks], axis=1)

        with slim.arg_scope(deepsets_arg_scope()):
            net, deepsets_end_points = deepsets(
                inputs=inputs,
                masks=masks,
                contexts=contexts,
                is_training=is_training,
                **deepsets_params)

        net = tf.sigmoid(net) # TODO: Maybe use a different final activation

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        if desc_encoder_inputs is not None:
            end_points.update(desc_end_points)
        if image_encoder_inputs is not None:
            end_points.update(image_end_points)
        end_points.update(deepsets_end_points)

        return net, end_points

