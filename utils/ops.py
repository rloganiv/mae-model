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
"""Useful TensorFlow operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def distance(x1, x2, distance_metric='euclidean'):
    """Computes the distance between rows.

    Args:
        x1: A tensor of size [m, k].
        x2: A tensor of size [m, k].
        distance_metric: Either 'euclidean' or 'cosine'.
    Returns:
        A tensor of size [m] whose elements are the distances between rows in
        x1 and x2.
    """
    if distance_metric == 'euclidean':
        return tf.reduce_sum((x1 - x2)**2, axis=1)
    elif distance_metric == 'cosine':
        x1 = tf.nn.l2_normalize(x1, axis=1)
        x2 = tf.nn.l2_normalize(x2, axis=1)
        return 1.0 - tf.reduce_sum(x1 * x2, axis=1)
    else:
        raise ValueError('Unknown distance distance_metric: %s' % distance_metric)


def distance_matrix(x1, x2, distance_metric='euclidean'):
    """Computes the pairwise distance between two tensors.

    Args:
        x1: A tensor of size [m, k].
        x2: A tensor of size [n, k].
        distance_metric: Either 'euclidean' or 'cosine'.
    Returns:
        A tensor of size [m, n] whose elements are the distances between x1 and
        x2.
    """
    if distance_metric == 'euclidean':
        l = tf.reduce_sum(x1**2, axis=1, keepdims=True)
        m = 2.0 * tf.tensordot(x1, x2, axes=[[1], [1]])
        r = tf.transpose(tf.reduce_sum(x2**2, axis=1, keepdims=True))
        return l - m + r
    elif distance_metric == 'cosine':
        x1 = tf.nn.l2_normalize(x1, axis=1)
        x2 = tf.nn.l2_normalize(x2, axis=1)
        return 1.0 - tf.tensordot(x1, x2, axes=[[1], [1]])
    else:
        raise ValueError('Unknown distance_metric: %s' % distance_metric)


def rank(scores, correct_value_ids):
    """Computes the rank of the correct value in a matrix of scores.

    Args:
        scores: A tensor of size [batch_size, num_values].
        correct_value_ids: A tensor of size [batch_size] whose i'th element is
            the integer corresponding to the correct value for the i'th example.

    Returns:
        A tensor of size [batch_size] whose i'th value is the rank of the
            correct value in the scores tensor.
    """
    batch_size = correct_value_ids.shape[0]
    indices = tf.stack([tf.range(batch_size), correct_value_ids], axis=1)
    correct = tf.gather_nd(scores, indices)
    correct = tf.expand_dims(correct, 1)
    incorrect = tf.cast(scores < correct, dtype=tf.float32)
    rank = tf.reduce_sum(incorrect, axis=1) + 1.0
    return rank

