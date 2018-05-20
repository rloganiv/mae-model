"""Provides utilities for sampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque, namedtuple
import json
import os
import pickle
import random

from .utils import Vocab, ValueSet


# TODO: Avoid hard-coding the path here!
with open('/home/rlogan/projects/mae-model/data/blank.jpeg', 'rb') as f:
    BLANK_IMAGE = f.read()


Product = namedtuple('Product', [
    'uri',
    'attr_query_id',
    'value_id',
    'desc_word_ids',
    'title_word_ids',
    'image_byte_strings',
    'known_attrs',
    'known_values'
])


def process_train(product,
                  config,
                  desc_vocab,
                  attr_vocab,
                  value_set):
    # URI.
    uri = product['diffbotUri']

    # Description.
    if config['model']['use_descs']:
        desc_word_ids = [desc_vocab.word2id(word) for word in
                         product['clean_text'].split()]
        if len(desc_word_ids) > config['training']['max_desc_length']:
            return []
    else:
        desc_word_ids = None

    # Title.
    if config['model']['use_titles']:
        title_word_ids = [desc_vocab.word2id(word) for word in
                         product['clean_title'].split()]
        if len(title_word_ids) > config['training']['max_desc_length']:
            return []
    else:
        title_word_ids = None

    # Images.
    if config['model']['use_images']:
        img_dir = config['data']['img_dir']
        image_byte_strings = []
        for fname in product['images']:
            fname = os.path.join(img_dir, fname)
            try:
                with open(fname, 'rb') as f:
                    image_byte_strings.append(f.read())
            except FileNotFoundError:
                continue
        if len(image_byte_strings) > config['training']['max_number_of_images']:
            return []
    else:
        image_byte_strings = None

    # Attribute Value pairs.
    av_pairs = list(product['specs'].items())
    attrs, values = zip(*av_pairs)
    attr_ids = [attr_vocab.word2id(x) for x in attrs]
    value_ids = [value_set.global_vocab.word2id(x) for x in values]

    # TODO: Fix
    known_attrs = []
    known_values = []

    # Construct output
    out = [Product(uri=uri,
                   attr_query_id=attr_query_id,
                   value_id=value_id,
                   desc_word_ids=desc_word_ids,
                   title_word_ids=title_word_ids,
                   image_byte_strings=image_byte_strings,
                   known_attrs=known_attrs,
                   known_values=known_values)
          for attr_query_id, value_id in zip(attr_ids, value_ids)]

    return out


def _pad(x, pad_value):
    max_len = max(len(i) for i in x)
    padded = [i + [pad_value]*(max_len - len(i)) for i in x]
    mask = [[1]*len(i) + [0]*(max_len - len(i)) for i in x]
    return padded, mask


def process_batch(batch, config):
    attr_query_ids = [x.attr_query_id for x in batch]
    value_ids =  [x.value_id for x in batch]
    uris = [x.uri for x in batch]
    feed_dict = {
        'attr_query_ids:0': attr_query_ids,
        'value_ids:0': value_ids,
    }

    if config['model']['use_descs']:
        desc_word_ids = [x.desc_word_ids for x in batch]
        desc_word_ids, desc_masks = _pad(desc_word_ids, pad_value=0)
        feed_dict['desc_word_ids:0'] = desc_word_ids
        feed_dict['desc_masks:0'] = desc_masks

    if config['model']['use_titles']:
        title_word_ids = [x.title_word_ids for x in batch]
        title_word_ids, title_masks = _pad(title_word_ids, pad_value=0)
        feed_dict['title_word_ids:0'] = title_word_ids
        feed_dict['title_masks:0'] = title_masks

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

    return feed_dict, uris


def filter(specs, attr_vocab, value_set):
    out = {}
    for attr, value in specs.items():
        if (attr in attr_vocab._word2id) and (value in value_set.global_vocab._word2id):
            out[attr] = value
    return out


def generate_batches(mode, config, fnames=None, mate=False, mc_attr=False):
    # Sanity checks.
    if mode not in ['train', 'val', 'val_gold', 'test']:
        raise ValueError('Bad mode: %s' % mode)

    # Get correct processing function.
    process_fn = process_train
    process_batch_fn = process_batch

    # Setup - get directories and filenames.
    img_dir = config['data']['img_dir']
    if mode == 'train':
        dir = config['data']['train_dir']
    elif mode == 'val':
        dir = config['data']['val_dir']
    elif mode == 'val_gold':
        dir =  config['data']['val_gold_dir']
    elif mode == 'test':
        fnames = fnames
    if fnames is None:
        fnames = [os.path.join(dir, fname) for fname in os.listdir(dir)]
    with open(config['data']['desc_file'], 'r') as f:
        desc_vocab = Vocab.load(f)
    with open(config['data']['attr_file'], 'r') as f:
        attr_vocab = Vocab.load(f)
    with open(config['data']['value_file'], 'r') as f:
        value_set = ValueSet.load(f)

    batch_size = config['training']['batch_size']

    # Main execution
    batch_queue = deque()
    while True:
        if mode == 'train':
           random.shuffle(fnames)
        for fname in fnames:
            print(fname)
            with open(fname, 'r') as f:
                products = json.load(f)
            if mode == 'train':
                random.shuffle(products)
            for product in products:
                # Deque batches if enough data in queue
                while len(batch_queue) >= batch_size:
                    batch = [batch_queue.pop() for _ in range(batch_size)]
                    yield process_batch_fn(batch, config)
                # Otherwise load more data
                product['specs'] = filter(product['specs'], attr_vocab,
                                          value_set)
                if len(product['specs']) == 0:
                    continue
                processed = process_fn(product, config, desc_vocab, attr_vocab,
                                       value_set)
                if config['model']['use_images']:
                    processed = [x for x in processed if len(x.image_byte_strings) != 0]
                batch_queue.extend(processed)
        if mode != 'train':
            break

