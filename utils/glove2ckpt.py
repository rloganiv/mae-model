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
"""Command-line utility for loading word embeddings into Tensorflow.

Usage:
    python glove2ckpt.py vectors.txt ckpt/glove.ckpt --vocab desc.txt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import random
import tensorflow as tf

from utils import Vocab


FLAGS = None


def main(_):
    if FLAGS.vocab:
        print('Loading vocab...')
        with open(FLAGS.vocab, 'r') as f:
            vocab = Vocab.load(f)

    print('Loading embeddings...')
    words = []
    embeddings = []
    with open(FLAGS.embedding_file, 'r') as f:
        for line in f:
            split = line.split()
            word = ' '.join(split[:-200])
            embedding = split[-200:]
            embedding = list(map(float, embedding))
            words.append(word)
            embeddings.append(embedding)
    embedding_size = len(embedding)
    if FLAGS.vocab:
        truncated_embeddings = []
        word2id = {w: i for i, w in enumerate(words)}
        for word in vocab._word2id:
            try:
                id = word2id[word]
                truncated_embeddings.append(embeddings[id])
            except KeyError:
                print('WARNING: Word "%s" has no predefined embedding' % word)
                random_embedding = [random.random() for _ in
                                    range(embedding_size)]
                truncated_embeddings.append(random_embedding)
        embedding_matrix = np.array(truncated_embeddings)
    else:
        embedding_matrix = np.array(embeddings)

    print('Producing Tensor:')
    embedding_matrix = tf.Variable(embedding_matrix,
                                   dtype=tf.float32,
                                   name='desc_word_embeddings')
    print(embedding_matrix)
    print('Saving checkpoint...')
    saver = tf.train.Saver([embedding_matrix])
    with tf.Session() as sess:
        sess.run(tf.variables_initializer([embedding_matrix]))
        saver.save(sess, FLAGS.output_file, write_meta_graph=False)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('embedding_file', type=str,
                        help='File containing the word embeddings.')
    parser.add_argument('output_file', type=str,
                        help='Output checkpoint file.')
    parser.add_argument('--vocab', type=str,
                        help='(Optional) File containing the vocabulary. '
                            'Only needed if target vocabulary does not '
                            'perfectly match that used in the embedding file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

