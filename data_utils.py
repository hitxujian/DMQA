# Modification of https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/rnn/translate/data_utils.py
#
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import time
import random
from tqdm import *
from glob import glob
from collections import defaultdict
import cPickle

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _UNK]

PAD_ID = 0
GO_ID = 1
UNK_ID = 2

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")



def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def dmqa_file_reader(dfile):
  with gfile.GFile(dfile, mode="r") as f:
    lines = f.read().split("\n\n")
    return lines


def load_dataset(data_dir, dataset_name, vocab_size, max_nsteps, part="training"):
  data = []
  data_path = os.path.join(data_dir, dataset_name, "questions", part)
  readed_data_path = os.path.join(data_dir, dataset_name, "%s_v%d_mn%d.pkl")
  if os.path.exists(readed_data_path):
    data = cPickle.load(open(readed_data_path))
  else:
    print("Load data from %s" %(data_path))
    for fname in tqdm(glob(os.path.join(data_path, "*.question.ids%s" % (vocab_size)))):
      try:
        tokens = dmqa_file_reader(fname)
        # check max_nsteps
        d = [int(t) for t in tokens[1].strip().split(' ')]
        q = [int(t) for t in tokens[2].strip().split(' ')]
        a = [int(tokens[3])]
        if len(d) + len(q) < max_nsteps:
          data.append((d,q,a))
      except Exception as e:
        print(" [!] Error occured for %s: %s" % (fname, e))
    cPickle.dump(data, open(readed_data_path, 'w'))
  return data

def batch_iter(data, batch_size, num_epochs, shuffle=True):
  """
  Generates a batch iterator for a dataset.
  """
  data_size = len(data)
  num_batches_per_epoch = int(len(data)/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    if shuffle:
      random.shuffle(data)
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield data[start_index:end_index]



def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Edit by Miao
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    for fname in tqdm(glob(os.path.join(data_path, "*.question"))):
      try:
        _, d, q, a, _ = dmqa_file_reader(fname)
        context = d + " " + q
        tokens = tokenizer(context) if tokenizer else basic_tokenizer(context)
        for w in tokens:
          word = _DIGIT_RE.sub("0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      except:
        print(" [!] Error occured for %s" % fname)

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocab,
                      tokenizer=None, normalize_digits=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in DMQA format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    try:
      results = dmqa_file_reader(data_path)
      with gfile.GFile(target_path, mode="w") as target_file:
        for i in range(5):
          if i == 0 or i == 4:
            target_file.write(results[i] + "\n\n")
          else:
            ids = sentence_to_token_ids(results[i], vocab, tokenizer,
                                            normalize_digits)
            target_file.write(" ".join(str(tok) for tok in ids) + "\n\n")
    except Exception as e:
      print(" [-] %s, %s" % (data_path, e))



def questions_to_token_ids(data_path, vocab_fname, vocab_size):
  vocab, _ = initialize_vocabulary(vocab_fname)
  for fname in tqdm(glob(os.path.join(data_path, "*.question"))):
    data_to_token_ids(fname, fname + ".ids%s" % vocab_size, vocab)


def prepare_data(data_dir, dataset_name, vocab_size):
  train_path = os.path.join(data_dir, dataset_name, 'questions', 'training')
  validation_path = os.path.join(data_dir, dataset_name, 'questions', 'validation')
  test_path = os.path.join(data_dir, dataset_name, 'questions', 'test')

  vocab_fname = os.path.join(data_dir, dataset_name, '%s.vocab%s' % (dataset_name, vocab_size))

  if not os.path.exists(vocab_fname):
    print(" [*] Create vocab from %s to %s ..." % (train_path, vocab_fname))
    create_vocabulary(vocab_fname, train_path, vocab_size)
  else:
    print(" [*] Skip creating vocab")

  print(" [*] Convert data in %s into vocab indicies..." % (train_path))
  questions_to_token_ids(train_path, vocab_fname, vocab_size)

  print(" [*] Convert data in %s into vocab indicies..." % (validation_path))
  questions_to_token_ids(validation_path, vocab_fname, vocab_size)

  print(" [*] Convert data in %s into vocab indicies..." % (test_path))
  questions_to_token_ids(test_path, vocab_fname, vocab_size)



if __name__ == '__main__':
  if len(sys.argv) < 3:
    print(" [*] usage: python data_utils.py DATA_DIR DATASET_NAME VOCAB_SIZE")
  else:
    data_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    if len(sys.argv) > 3:
      vocab_size = sys.argv[3]
    else:
      vocab_size = 100000

    prepare_data(data_dir, dataset_name, int(vocab_size))
