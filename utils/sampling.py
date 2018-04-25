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
    'correct_value_id',
    'incorrect_value_id',
    'desc_word_ids',
    'title_word_ids',
    'image_byte_strings',
    'known_attrs',
    'known_values'
])

MATEProduct = namedtuple('Product', [
    'pos_attr_query_id',
    'neg_attr_query_id',
    'desc_word_ids',
    'image_byte_strings',
    'known_attrs',
])

MCAttrProduct = namedtuple('Product', [
    'uri',
    'attr_ids',
    'desc_word_ids',
    'image_byte_strings',
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
            uri=uri,
            attr_query_id=attr_query_id,
            correct_value_id=correct_value_id,
            incorrect_value_id=incorrect_value_id,
            desc_word_ids=desc_word_ids,
            title_word_ids=title_word_ids,
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
            uri=uri,
            attr_query_id=attr_query_id,
            correct_value_id=correct_value_id,
            incorrect_value_id=incorrect_value_id,
            desc_word_ids=desc_word_ids,
            title_word_ids=title_word_ids,
            image_byte_strings=image_byte_strings,
            known_attrs=known_attrs,
            known_values=known_values)
        out.append(product)

    if config['training']['neg_sample_unk']:
        product = Product(
            uri=uri,
            attr_query_id=attr_query_id,
            correct_value_id=correct_value_id,
            incorrect_value_id=unk_value_id,
            desc_word_ids=desc_word_ids,
            title_word_ids=title_word_ids,
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
                uri=uri,
                attr_query_id=unk_attr_id,
                correct_value_id=unk_value_id,
                incorrect_value_id=incorrect_value_id,
                desc_word_ids=desc_word_ids,
                title_word_ids=title_word_ids,
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
                uri=uri,
                attr_query_id=unk_attr_id,
                correct_value_id=unk_value_id,
                incorrect_value_id=incorrect_value_id,
                desc_word_ids=desc_word_ids,
                title_word_ids=title_word_ids,
                image_byte_strings=image_byte_strings,
                known_attrs=known_attrs,
                known_values=known_values)
            out.append(product)

    return out


def process_test(product,
                 config,
                 desc_vocab,
                 attr_vocab,
                 value_set):
    # Description.
    if config['model']['use_descs']:
        desc_word_ids = [desc_vocab.word2id(word) for word in
                         product['clean_text'].split()]
        if len(desc_word_ids) > config['training']['max_desc_length']:
            return []
    else:
        desc_word_ids = None

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

    # Add all attributes.
    out = []
    incorrect_value_id = 0 # Arbitrary
    known_attrs, known_values = zip(*list(product['specs'].items()))
    known_attrs = [attr_vocab.word2id(attr) for attr in known_attrs]
    known_values = [value_set.global_vocab.word2id(value) for value in known_values]
    for attr in attr_vocab._id2word:
        attr_query_id = attr_vocab.word2id(attr)
        if attr in product['specs']: # Non-unk
            correct_value_id = value_set.global_vocab.word2id(product['specs'][attr])
            tmp_known_attrs = [attr for attr in known_attrs if attr != attr_query_id]
            tmp_known_values = [value for value in known_values if value != correct_value_id]
            product_out = Product(
                attr_query_id=attr_query_id,
                correct_value_id=correct_value_id,
                incorrect_value_id=incorrect_value_id,
                desc_word_ids=desc_word_ids,
                image_byte_strings=image_byte_strings,
                known_attrs=tmp_known_attrs,
                known_values=tmp_known_values)
            out.append(product_out)
        else: # Unk
            correct_value_id = len(value_set.global_vocab)
            product_out = Product(
                attr_query_id=attr_query_id,
                correct_value_id=correct_value_id,
                incorrect_value_id=incorrect_value_id,
                desc_word_ids=desc_word_ids,
                image_byte_strings=image_byte_strings,
                known_attrs=known_attrs,
                known_values=known_values)
            out.append(product_out)
    return out


def process_mate(product,
                 config,
                 desc_vocab,
                 attr_vocab,
                 value_set):
    # Description.
    if config['model']['use_descs']:
        desc_word_ids = [desc_vocab.word2id(word) for word in
                         product['clean_text'].split()]
        if len(desc_word_ids) > config['training']['max_desc_length']:
            return []
    else:
        desc_word_ids = None

    # Images.
    if config['model']['use_images']:
        img_dir = config['data']['img_dir']
        image_byte_strings = []
        for fname in product['images']:
            fname = os.path.join(img_dir, fname)
            with open(fname, 'rb') as f:
                image_byte_strings.append(f.read())
        if len(image_byte_strings) > config['training']['max_number_of_images']:
            return []
    else:
        image_byte_strings = None

    # Attribute Value pairs.
    av_pairs = list(product['specs'].items())
    attrs, values = zip(*av_pairs)
    attr_ids = [attr_vocab.word2id(x) for x in attrs]
    index = random.randrange(len(av_pairs))
    pos_attr_query = attrs[index]
    pos_attr_query_id = attr_ids[index]
    known_attrs = attr_ids[:index] + attr_ids[index+1:]

    # Randomly select subset of known attributes.
    size = random.randint(0, len(known_attrs))
    random.shuffle(known_attrs)
    known_attrs = known_attrs[:size]

    # Construct output

    method = config['training']['sampling_method']
    neg_attr_query_id = pos_attr_query_id
    while neg_attr_query_id in attr_ids:
        neg_attr_query_id = attr_vocab.sample(method=method)
    processed_product = MATEProduct(
        pos_attr_query_id=pos_attr_query_id,
        neg_attr_query_id=neg_attr_query_id,
        desc_word_ids=desc_word_ids,
        image_byte_strings=image_byte_strings,
        known_attrs=known_attrs)

    return [(processed_product, product)] # Stupid hack, needed for eval


def process_mc_attr(product,
                    config,
                    desc_vocab,
                    attr_vocab,
                    value_set):
    # URI.
    uri = product['diffbotUri']

    # Attribute ids.
    attr_ids = [attr_vocab.word2id(x) for x in product['specs'].keys()]

    # Description.
    if config['model']['use_descs']:
        desc_word_ids = [desc_vocab.word2id(word) for word in
                         product['clean_text'].split()]
        if len(desc_word_ids) > config['training']['max_desc_length']:
            return []
    else:
        desc_word_ids = None

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

    return [MCAttrProduct(uri=uri,
                          attr_ids=attr_ids,
                          desc_word_ids=desc_word_ids,
                          image_byte_strings=image_byte_strings)]


def _pad(x, pad_value):
    max_len = max(len(i) for i in x)
    padded = [i + [pad_value]*(max_len - len(i)) for i in x]
    mask = [[1]*len(i) + [0]*(max_len - len(i)) for i in x]
    return padded, mask


def process_batch(batch, config):
    attr_query_ids = [x.attr_query_id for x in batch]
    correct_value_ids =  [x.correct_value_id for x in batch]
    incorrect_value_ids = [x.incorrect_value_id for x in batch]
    uris = [x.uri for x in batch]
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
    # TODO: Uncomment
    # return feed_dict
    return feed_dict, uris


def process_batch_mate(batch, config):
    batch, products = zip(*batch)
    pos_attr_query_ids = [x.pos_attr_query_id for x in batch]
    neg_attr_query_ids = [x.neg_attr_query_id for x in batch]
    feed_dict = {
        'pos_attr_query_ids:0': pos_attr_query_ids,
        'neg_attr_query_ids:0': neg_attr_query_ids
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

    known_attrs = [x.known_attrs for x in batch]
    known_attrs, table_masks = _pad(known_attrs, pad_value=0)
    feed_dict['known_attrs:0'] = known_attrs
    feed_dict['table_masks:0'] =  table_masks

    return feed_dict, products


def process_batch_mc_attr(batch, config):
    uris = [x.uri for x in batch]

    attr_ids = [x.attr_ids for x in batch]
    attr_ids, _ = _pad(attr_ids, pad_value=-1)
    feed_dict = {
        'attr_ids:0': attr_ids,
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

    return feed_dict, uris


def filter(specs, attr_vocab, value_set):
    out = {}
    for attr, value in specs.items():
        if (attr in attr_vocab._word2id) and (value in value_set.global_vocab._word2id):
            out[attr] = value
    return out


def generate_batches(mode, config, mate=False, mc_attr=False):
    # Sanity checks.
    if mode not in ['train', 'val', 'test']:
        raise ValueError('Bad mode: %s' % mode)

    # Get correct processing function.
    # TODO: Uncomment
    # if mode == 'test':
    #     process_fn = process_test
    # else:
    #     process_fn = process_train
    process_fn = process_train

    if mate:
        process_fn = process_mate
        process_batch_fn = process_batch_mate
    else:
        process_batch_fn = process_batch

    # TODO: Clean up!!!
    if mc_attr:
        process_fn = process_mc_attr
        process_batch_fn = process_batch_mc_attr

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

    batch_size = config['training']['batch_size']

    # Main execution
    batch_queue = deque()
    while True:
        if mode == 'train':
           random.shuffle(fnames)
        for fname in fnames:
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

