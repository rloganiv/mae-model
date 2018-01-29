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

from collections import Counter, deque, namedtuple
import json
import os
import random
import yaml


# TODO: Avoid hard-coding the path here
with open('data/img/blank.jpeg', 'rb') as f:
    BLANK_IMAGE = f.read()


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
        try:
            total = sum(self._counts.values())
        except:
            import pdb; pdb.set_trace()
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


Product = namedtuple('Product', [
    'attr_query_id',
    'correct_value_id',
    'incorrect_value_id',
    'desc_word_ids',
    'image_byte_strings',
    'known_attrs',
    'known_values'
])


def process_for_training(product,
                         config,
                         desc_vocab,
                         attr_vocab,
                         value_set):
    # Description.
    desc_word_ids = [desc_vocab.word2id(word) for word in product['tokens']]
    if len(desc_word_ids) > config['training']['max_desc_length']:
        return []

    # Images.
    img_dir = config['data']['img_dir']
    image_byte_strings = []
    for fname in product['images']:
        fname = os.path.join(img_dir, fname)
        with open(fname, 'rb') as f:
            image_byte_strings.append(f.read())
    if len(image_byte_strings) > config['training']['max_number_of_images']:
        return []

    # Attribute Value pairs.
    av_pairs = list(product['specs'].items())
    attrs, values = zip(*av_pairs)
    attr_ids = [attr_vocab.word2id(x) for x in attrs]
    value_ids = [value_set.global_vocab.word2id(x) for x in values]
    index = random.randrange(len(av_pairs))
    attr_query = attrs[index]
    attr_query_id = attr_ids[index]
    correct_value_id = value_ids[index]
    known_attrs = attr_ids[:index] + attr_ids[index+1:]
    known_values = value_ids[:index] + value_ids[index+1:]
    unk_value_id = len(value_set.global_vocab)

    # Construct output
    out = []

    method = config['training']['sampling_method']
    if config['training']['neg_sample_from_all_values']:
        incorrect_value_id = correct_value_id
        while incorrect_value_id == correct_value_id:
            incorrect_value_id = value_set.sample(method=method)
        product = Product(
            attr_query_id=attr_query_id,
            correct_value_id=correct_value_id,
            incorrect_value_id=incorrect_value_id,
            desc_word_ids=desc_word_ids,
            image_byte_strings=image_byte_strings,
            known_attrs=known_attrs,
            known_values=known_values)
        out.append(product)

    if config['training']['neg_sample_from_attr_values']:
        incorrect_value_id = correct_value_id
        while incorrect_value_id == correct_value_id:
            incorrect_value_id = value_set.sample(attr_query,
                                                  method=method)
        product = Product(
            attr_query_id=attr_query_id,
            correct_value_id=correct_value_id,
            incorrect_value_id=incorrect_value_id,
            desc_word_ids=desc_word_ids,
            image_byte_strings=image_byte_strings,
            known_attrs=known_attrs,
            known_values=known_values)
        out.append(product)

    if config['training']['neg_sample_unk']:
        product = Product(
            attr_query_id=attr_query_id,
            correct_value_id=correct_value_id,
            incorrect_value_id=unk_value_id,
            desc_word_ids=desc_word_ids,
            image_byte_strings=image_byte_strings,
            known_attrs=known_attrs,
            known_values=known_values)
        out.append(product)

    if config['training']['pos_sample_unk']:
        unk_attr_id = attr_query_id
        while unk_attr_id == attr_query_id:
            unk_attr_id = attr_vocab.sample(method)
            unk_attr = attr_vocab.id2word(unk_attr_id)

        if config['training']['neg_sample_from_all_values']:
            incorrect_value_id = correct_value_id
            while incorrect_value_id == correct_value_id:
                incorrect_value_id = value_set.sample(method=method)
            product = Product(
                attr_query_id=unk_attr_id,
                correct_value_id=unk_value_id,
                incorrect_value_id=incorrect_value_id,
                desc_word_ids=desc_word_ids,
                image_byte_strings=image_byte_strings,
                known_attrs=known_attrs,
                known_values=known_values)
            out.append(product)

        if config['training']['neg_sample_from_attr_values']:
            incorrect_value_id = correct_value_id
            while incorrect_value_id == correct_value_id:
                incorrect_value_id = value_set.sample(unk_attr,
                                                      method=method)
            product = Product(
                attr_query_id=unk_attr_id,
                correct_value_id=unk_value_id,
                incorrect_value_id=incorrect_value_id,
                desc_word_ids=desc_word_ids,
                image_byte_strings=image_byte_strings,
                known_attrs=known_attrs,
                known_values=known_values)
            out.append(product)

    return out


def process_for_evaluation(item, img_dir, mappers):
    raise NotImplementedError


def _pad(x, pad_value):
    max_len = max(len(i) for i in x)
    padded = [i + [pad_value]*(max_len - len(i)) for i in x]
    mask = [[1]*len(i) + [0]*(max_len - len(i)) for i in x]
    return padded, mask


def process_batch(batch, config):
    attr_query_ids = [x.attr_query_id for x in batch]
    correct_value_ids =  [x.correct_value_id for x in batch]
    incorrect_value_ids = [x.incorrect_value_id for x in batch]
    feed_dict = {
        'attr_query_ids:0': attr_query_ids,
        'correct_value_ids:0': correct_value_ids,
        'incorrect_value_ids:0': incorrect_value_ids,
    }

    if config['model']['use_descs']:
        desc_word_ids = [x.desc_word_ids for x in batch]
        desc_word_ids, desc_masks = _pad(desc_word_ids, pad_value=0)
        feed_dict['desc_word_ids:0'] = desc_word_ids
        feed_dict['desc_masks:0'] = desc_masks

    if config['model']['use_images']:
        image_byte_strings = [x.image_byte_strings for x in batch]
        image_byte_strings, image_masks = _pad(image_byte_strings,
                                               pad_value=BLANK_IMAGE)
        feed_dict['image_byte_strings:0'] = image_byte_strings
        feed_dict['image_masks:0'] = image_masks

    if config['model']['use_tables']:
        known_attrs = [x.known_attrs for x in batch]
        known_attrs, table_masks = _pad(known_attrs, pad_value=0)
        known_values = [x.known_values for x in batch]
        known_values, _ = _pad(known_values, pad_value=0)
        feed_dict['known_attrs:0'] = known_attrs
        feed_dict['known_values:0'] = known_values
        feed_dict['table_masks:0'] =  table_masks

    return feed_dict


def generate_batches(mode, config):
    # Sanity checks.
    if mode not in ['train', 'val', 'test']:
        raise ValueError('Bad mode: %s' % mode)

    # Setup - get directories and filenames.
    img_dir = config['data']['img_dir']
    if mode == 'train':
        dir = config['data']['train_dir']
    elif mode == 'val':
        dir = config['data']['val_dir']
    elif mode == 'test':
        dir = config['data']['test_dir']
    fnames = [os.path.join(dir, fname) for fname in os.listdir(dir)]
    with open(config['data']['desc_file'], 'r') as f:
        desc_vocab = Vocab.load(f)
    with open(config['data']['attr_file'], 'r') as f:
        attr_vocab = Vocab.load(f)
    with open(config['data']['value_file'], 'r') as f:
        value_set = ValueSet.load(f)

    if mode == 'train':
        epochs = config['training']['epochs']
        batch_size = config['training']['batch_size']
        process_fn = process_for_training
    else:
        epochs = 1
        batch_size = 1
        process_fn = process_for_evaluation

    # Main execution
    batch = []
    for i in range(epochs):
        if mode == 'train':
           random.shuffle(fnames)
        for fname in fnames:
            with open(fname, 'r') as f:
                products = json.load(f)
            if mode == 'train':
                random.shuffle(products)
            for product in products:
                if len(batch) >= batch_size:
                    yield process_batch(batch[:batch_size], config)
                    batch = batch[batch_size:]
                processed = process_fn(product, config, desc_vocab, attr_vocab,
                                       value_set)
                batch.extend(processed)

