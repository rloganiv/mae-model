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
"""Contains the model definition for the Multiclass Attribute Extraction model.

The model definition was introduces in the following paper:

    To be accepted by a major conference/journal

@@multiclass_attr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers

from nets.desc_encoder import desc_encoder
from nets.image_encoder import image_encoder

slim = tf.contrib.slim


def multiclass_attr(num_outputs=1,
                    hidden_size=1024,
                    image_encoder_inputs=None,
                    image_encoder_masks=None,
                    image_encoder_params={},
                    desc_encoder_inputs=None,
                    desc_encoder_masks=None,
                    desc_encoder_params={},
                    is_training=False,
                    reuse=None,
                    scope=None):
    """Multiclass attribute extraction architecture.

    Args:
        num_outputs: Size of network output.
        image_encoder_inputs: Tensor of size [batch_size, n_images, 224, 244, 3] to be
            fed into image encoder architecture.
        image_encoder_masks: Tensor of size [batch_size, n_images] to mask padded
            values in image_inputs.
        image_encoder_params: Dict of parameters to be used in image encoder.
        desc_encoder_inputs: Tensor of size [] to be fed into desc encoder
            architecture.
        desc_encoder_masks: #TODO: Document.
        desc_encoder_params: Dict of parameters to be used in desc encoder.
        fusion_method: One of 'concat' or 'attention'.
        reuse: Whether or not the network and its variables should be reused.
            Requires that 'scope' is given.
        scope: Optional scope for the variables.

    Returns:
        net: The output of the network.
        end_points: A dict of tensors with intermediate activations.
    """
    model_inputs = [
        desc_encoder_inputs,
        desc_encoder_masks,
        image_encoder_inputs,
        image_encoder_masks,
    ]

    with tf.variable_scope(scope, 'multiclass_attr', model_inputs, reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        branches = []

        if desc_encoder_inputs is not None:
            if desc_encoder_masks is None:
                raise ValueError('desc_encoder_masks must be specified.')
            desc_encoding, desc_end_points = desc_encoder(
                desc_encoder_inputs,
                desc_encoder_masks,
                is_training=is_training,
                **desc_encoder_params)
            branches.append(desc_encoding)

        if image_encoder_inputs is not None:
            if image_encoder_masks is None:
                raise ValueError('image_encoder_masks must be specified.')
            image_encoding, image_end_points = image_encoder(
                image_encoder_inputs,
                image_encoder_masks,
                is_training=is_training,
                **image_encoder_params)
            branches.append(image_encoding)

        if len(branches) == 1:
            net = branches[0]
        else:
            net = tf.concat(branches, axis=1)
            net = slim.fully_connected(net, num_outputs=hidden_size)

        # Final FC layer - note: outputs are logits not sigmoids!
        net = slim.fully_connected(net,
                                   num_outputs=num_outputs,
                                   activation_fn=None)

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)

        if desc_encoder_inputs is not None:
            end_points.update(desc_end_points)
        if image_encoder_inputs is not None:
            end_points.update(image_end_points)

        return net, end_points

