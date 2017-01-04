import os
import numpy as np
import tensorflow as tf
import datetime

#from model import DeepLSTM, DeepBiLSTM, AttentiveReader
from model import DeepLSTM

from utils import pp
import data_utils

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("vocab_size", 100000, "The size of vocabulary [10000]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_integer("cell_size", 256, "The cell size of  rnn [256]")
flags.DEFINE_integer("max_nsteps", 1000, "Max nsteps for rnn [1000]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate [0.0001]")
flags.DEFINE_float("momentum", 0.9, "Momentum of RMSProp [0.9]")
flags.DEFINE_float("decay", 0.95, "Decay of RMSProp [0.95]")
flags.DEFINE_float("dropout_keep_prob", 0.1, "Dropout keep probability (default: 0.1)")
flags.DEFINE_string("model", "LSTM", "The type of model to train and test [LSTM, BiLSTM, Attentive, Impatient]")
flags.DEFINE_string("data_dir", "data", "The name of data directory [data]")
flags.DEFINE_string("dataset", "cnn", "The name of dataset [cnn, dailymail]")
flags.DEFINE_string("model_dir", "", "Directory name to save the model (summaries and checkpoints)")

FLAGS = flags.FLAGS

model_dict = {
  'LSTM': DeepLSTM
}
'''
model_dict = {
  'LSTM': DeepLSTM,
  'BiLSTM': DeepBiLSTM,
  'Attentive': AttentiveReader,
  'Impatient': None,
}
'''

gpu=0

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not FLAGS.model_dir:
    print(" [-] Error: Model dir is not set!") 
    exit(-1)

  if not os.path.exists(FLAGS.model_dir):
    print(" [*] Creating model directory...")
    os.makedirs(FLAGS.model_dir)

  with open(os.path.join(FLAGS.model_dir, "config.json"), 'w') as config_file:
    config_file.write("%s" %(pp.pformat(flags.FLAGS.__flags)))

  # build model
  model = model_dict[FLAGS.model](vocab_size = FLAGS.vocab_size, size=FLAGS.cell_size)
  # load data
  print(" [*] Loading dataset...")
  train_data = data_utils.load_dataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.vocab_size, FLAGS.max_nsteps, part="training")
  dev_data = data_utils.load_dataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.vocab_size, FLAGS.max_nsteps, part="validation")
  print(" [+] Finish loading. Train set: %d, Dev set: %d" %(len(train_data), len(dev_data)))

  model.train(train_data, dev_data, nb_epoch=FLAGS.epoch, batch_size=FLAGS.batch_size, model_dir=FLAGS.model_dir)
if __name__ == '__main__':
  tf.app.run()
