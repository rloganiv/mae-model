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
"""Command-line utility for extracting the attribute vocabulary, description
vocabulary and value sets from JSON data.

Usage:
    python json_preprocessing.py data data/json/train/*
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter, defaultdict
import argparse
import json
import os
import pickle

from utils import Vocab, ValueSet


FLAGS = None


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        print('Creating directory: %s' % FLAGS.output_dir)
        os.mkdir(FLAGS.output_dir)

    desc_counter = Counter()
    attr_counter = Counter()
    partial_counts = defaultdict(Counter)

    print('Processing data...')
    n = len(FLAGS.inputs)
    for i, fname in enumerate(FLAGS.inputs):
        print('File %i of %i: %s' % (i, n, fname))
        with open(fname, 'r') as f:
            data = json.load(f)
        for product in data:
            desc = product['tokens']
            desc_counter.update(desc)
            for attr, value in product['specs'].items():
                attr_counter.update((attr,))
                partial_counts[attr].update((value,))

    # Filter values
    partial_counts = {x: {y: z for y, z in y.items() if z > FLAGS.min_value }
                      for x, y in partial_counts.items()}

    # Remove singular attributes
    singular = {x for x, y in partial_counts.items() if len(y) <= 1}
    attr_counter = Counter({x: y for x, y in attr_counter.items() if x not in singular})
    partial_counts = {x: y for x, y in partial_counts.items() if x not in singular}

    # Filter attrs
    if FLAGS.max_attr is not None:
        attr_counter = {x: y for x, y in attr_counter.most_common(FLAGS.max_attr)}

    # Filter desc
    desc_counter = Counter({x: y for x, y in desc_counter.items() if y >= FLAGS.min_desc})
    desc_vocab = Vocab.build_from_counter(desc_counter)
    attr_vocab = Vocab.build_from_counter(attr_counter)
    value_set = ValueSet.build_from_partial_counts(partial_counts)

    print('Writing to disk...')
    desc_fname = os.path.join(FLAGS.output_dir, 'desc.txt')
    with open(desc_fname, 'w') as f:
        desc_vocab.write(f)
    attr_fname = os.path.join(FLAGS.output_dir, 'attr.txt')
    with open(attr_fname, 'w') as f:
        attr_vocab.write(f)
    value_fname = os.path.join(FLAGS.output_dir, 'value.txt')
    with open(value_fname, 'w') as f:
        value_set.write(f)
    stats_fname = os.path.join(FLAGS.output_dir, 'stats.txt')
    with open(stats_fname, 'w') as f:
        f.write('num_attrs: %i\n' % len(attr_vocab))
        f.write('num_vals: %i\n' % len(value_set))
        f.write('num_words: %i\n' % len(desc_vocab))

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Ouput directory.')
    parser.add_argument('inputs', type=str, nargs='+', help='Input files.')
    parser.add_argument('--max_attr', type=int, default=None,
                        help='Maximum number of attributes.')
    parser.add_argument('--min_value', type=int, default=1,
                        help='Minimum number of times a value must occur.')
    parser.add_argument('--min_desc', type=int, default=1,
                        help='Minimum number of times a word must occur.')
    parser.add_argument('--attr_map', type=str, default=None,
                        help='Path to attribute map file.')
    parser.add_argument('--value_map', type=str, default=None,
                        help='Path to value map file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

