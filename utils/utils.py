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
"""Provides utilities for Multimodal Attribute Extraction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import json
import random
import yaml


def load_config(fname):
    """Loads a configuation file.

    Args:
        fname: Name of the configuration file.

    Returns:
        An object containing the configuration parameters.
    """
    with open(fname, 'r') as f:
        config = yaml.load(f)
    return config


class Vocab(object):

    def __init__(self, unk_token='<UNK>'):
        self._unk_token = unk_token
        self._counts = None
        self._id2word = None
        self._word2id = None

    def __len__(self):
        return len(self._id2word)

    @classmethod
    def load(cls, f):
        """Loads vocabulary from a text file."""
        vocab = cls()
        vocab._id2word = []
        vocab._counts = Counter()
        for line in f:
            word, count = line.strip().split('\t')
            vocab._id2word.append(word)
            vocab._counts[word] = int(count)
        vocab._word2id = {word: i for i, word in enumerate(vocab._id2word)}
        return vocab

    def write(self, f):
        """Writes vocabulary to a text file."""
        for word in self._id2word:
            line = '%s\t%i\n' % (word, self._counts[word])
            f.write(line)

    @classmethod
    def build_from_counter(cls, counter):
        """Builds vocabulary from a collections.Counter object."""
        vocab = cls()
        vocab._counts = counter
        vocab._id2word = sorted(counter, key=counter.get, reverse=True)
        vocab._word2id = {word: i for i, word in enumerate(vocab._id2word)}
        return vocab

    def word2id(self, word):
        """Looks up the id of a word in vocabulary."""
        if word in self._word2id:
            return self._word2id[word]
        else:
            return len(self)

    def id2word(self, id):
        """Looks up the word with a given id from the vocabulary."""
        if id == len(self):
            return self._unk_token
        else:
            return self._id2word[id]

    def sample(self, method='unigram'):
        """Sample word id from the vocabulary.

        Args:
            method: Either 'uniform', 'unigram'.
        """
        total = sum(self._counts.values())
        if method == 'unigram':
            rng = random.randint(1, total)
            accumulated = 0
            for i, x in enumerate(self._counts.values()):
                accumulated += x
                if accumulated >= rng:
                    return i
        elif method == 'uniform':
            return random.randrange(len(self))
        else:
            raise ValueError('Bad sampling method %s' % method)


class ValueSet(object):

    def __init__(self, unk_token='<UNK>'):
        self._unk_token = unk_token
        self.partial_vocabs = None
        self.global_vocab = None

    def __len__(self):
        return len(self.global_vocab)

    @classmethod
    def load(cls, f):
        """Loads value set from a text file."""
        partial_vocabs = {}
        total_counter = Counter()
        data = json.load(f)
        for attr, value_dict in data.items():
            counter = Counter(value_dict)
            partial_vocabs[attr] = Vocab.build_from_counter(counter)
            total_counter += counter

        global_vocab = Vocab.build_from_counter(total_counter)
        value_set = cls()
        value_set.partial_vocabs = partial_vocabs
        value_set.global_vocab = global_vocab
        return value_set

    def write(self, f):
        """Writes value set to a text file."""
        d = {}
        for attr, vocab in self.partial_vocabs.items():
            d[attr] = vocab._counts
        json.dump(d, f, indent=1)

    @classmethod
    def build_from_partial_counts(cls, partial_counts):
        """Builds value set from a dictionary whose keys are attributes and
        whose values are Counters storing the counts of the values associated
        to the attribute.
        """
        partial_vocabs = {}
        total_counter = Counter()
        for attr, counter in partial_counts.items():
            partial_vocabs[attr] = Vocab.build_from_counter(counter)
            total_counter += counter
        global_vocab = Vocab.build_from_counter(total_counter)
        value_set = cls()
        value_set.partial_vocabs = partial_vocabs
        value_set.global_vocab = global_vocab
        return value_set

    def sample(self, attr=None, method='unigram'):
        if attr:
            tmp_id = self.partial_vocabs[attr].sample(method)
            word = self.partial_vocabs[attr].id2word(tmp_id)
            value_id = self.global_vocab.word2id(word)
        else:
            value_id = self.global_vocab.sample(method)
        return value_id


