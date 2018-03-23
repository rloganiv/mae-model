import argparse
import json
import pickle
import utils

from collections import defaultdict

FLAGS = None


def main(_):
    with open(FLAGS.attr_vocab, 'r') as f:
        attr_vocab = utils.Vocab.load(f)
    with open(FLAGS.value_vocab, 'r') as f:
        value_set = utils.ValueSet.load(f)
    if FLAGS.attr_map is not None:
        with open(FLAGS.attr_map, 'rb') as f:
            attr_map = pickle.load(f)
    if FLAGS.value_map is not None:
        with open(FLAGS.value_map, 'rb') as f:
            value_map = pickle.load(f)


    top_1 = 0
    top_20 = 0
    total = 0

    for fname in FLAGS.inputs:
        with open(fname, 'r') as f:
            products = json.load(f)
        for product in products:
            for attr, correct_value in product['specs'].items():
                try:
                    attr = attr_map[attr]
                    correct_value = value_map[correct_value]
                    partial_vocab = value_set.partial_vocabs[attr]
                except KeyError:
                    continue
                counts = partial_vocab._counts.most_common(20)
                preds = [x[0] for x in counts]
                total += 1
                if correct_value == preds[0]:
                    top_1 += 1
                if correct_value in preds:
                    top_20 += 1

    print('top 1: %0.4f' % (top_1 / total))
    print('top 20: %0.4f' % (top_20 / total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=str, nargs='+')
    parser.add_argument('--attr_vocab', type=str, required=True)
    parser.add_argument('--value_vocab', type=str, required=True)
    parser.add_argument('--attr_map', type=str, default=None)
    parser.add_argument('--value_map', type=str, default=None)
    FLAGS, _ = parser.parse_known_args()

    main(_)

