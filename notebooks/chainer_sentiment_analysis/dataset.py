# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import csv
import glob
import io
import os
import shutil
import tarfile
import tempfile

import numpy

import chainer

from code.nlp_utils import transform_to_array, split_text, normalize_text, make_vocab

URL_IMDB = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'


def download_imdb():
    path = chainer.dataset.cached_download(URL_IMDB)
    tf = tarfile.open(path, 'r')
    # To read many files fast, tarfile is untared
    path = tempfile.mkdtemp()
    tf.extractall(path)
    return path


def get_imdb(vocab=None, shrink=1, fine_grained=False,
             char_based=False):
    tmp_path = download_imdb()

    print('read imdb')
    train = read_imdb(tmp_path, 'train',
                      shrink=shrink, fine_grained=fine_grained,
                      char_based=char_based)
    test = read_imdb(tmp_path, 'test',
                     shrink=shrink, fine_grained=fine_grained,
                     char_based=char_based)

    shutil.rmtree(tmp_path)

    if vocab is None:
        print('constract vocabulary based on frequency')
        vocab = make_vocab(train)

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab


def read_imdb(path, split,
              shrink=1, fine_grained=False, char_based=False):
    fg_label_dict = {'1': 0, '2': 0, '3': 1, '4': 1,
                     '7': 2, '8': 2, '9': 3, '10': 3}

    def read_and_label(posneg, label):
        dataset = []
        target = os.path.join(path, 'aclImdb', split, posneg, '*')
        for i, f_path in enumerate(glob.glob(target)):
            if i % shrink != 0:
                continue
            with io.open(f_path, encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            tokens = split_text(normalize_text(text), char_based)
            if fine_grained:
                # extract from f_path. e.g. /pos/200_8.txt -> 8
                label = fg_label_dict[f_path.split('_')[-1][:-4]]
                dataset.append((tokens, label))
            else:
                dataset.append((tokens, label))
        return dataset

    pos_dataset = read_and_label('pos', 0)
    neg_dataset = read_and_label('neg', 1)
    return pos_dataset + neg_dataset
