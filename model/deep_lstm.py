# coding=utf-8
import time
import random
import numpy as np
import tensorflow as tf

from base_model import Model
from cells import LSTMCell, MultiRNNCellWithSkipConn

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import data_utils


class DeepLSTM(Model):
  """Deep LSTM model."""
  def __init__(self,
               size=256,
               vocab_size=100000,
               depth=2,
               batch_size=32,
               max_nsteps=1000,
               qca=True):
    """Initialize the parameters for an Deep LSTM model.
    
    Args:
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      keep_prob: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      max_nsteps: int, The max time unit [1000]
      qca: if true: the input is q+d, else: the input is d+q
    """
    super(DeepLSTM, self).__init__()

    self.size = int(size)
    self.vocab_size = int(vocab_size)
    self.depth = int(depth)
    #self.batch_size = int(batch_size)
    self.output_size = self.depth * self.size
    self.max_nsteps = int(max_nsteps)
    self.qca = qca

    # Placeholders for input, output and dropout
    self.input_x = tf.placeholder(tf.int32, [None, self.max_nsteps], name="input_x")
    self.input_y = tf.placeholder(tf.int32, [None, self.vocab_size], name="input_y")
    self.mask_y = tf.placeholder(tf.int32, [None, self.vocab_size], name="mask_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    start = time.clock()
    print(" [*] Building Deep LSTM...")

    real_batch_size = tf.shape(self.input_x)[0]
    print real_batch_size
    self.cell = LSTMCell(size)
    self.stacked_cell = MultiRNNCellWithSkipConn([self.cell] * depth)
    self.initial_state = self.stacked_cell.zero_state(real_batch_size, tf.float32)

    self.emb = tf.get_variable("emb", [self.vocab_size, self.size])
    #tf.summary.histogram("embed", self.emb)
    embed_inputs = tf.nn.embedding_lookup(self.emb, tf.transpose(self.input_x))
    
    _, states = tf.nn.rnn(self.stacked_cell,
                        tf.unpack(embed_inputs),
                        dtype=tf.float32,
                        initial_state=self.initial_state)
    self.states = tf.pack(states)
    state_layers = tf.split(1, self.depth, self.states)
    hiddens = []
    for layer in state_layers:
      h,c,y = tf.split(1, 3, layer)
      hiddens.append(y)
    self.feature = tf.concat(1, hiddens, name="feature")
    self.f_drop = tf.nn.dropout(self.feature, self.dropout_keep_prob, name="f_drop")

    with tf.variable_scope('output'):
      W = tf.get_variable(
          "W",
          shape=[self.output_size, self.vocab_size],
          initializer=tf.contrib.layers.xavier_initializer()) 
      
      #self.y_ = tf.matmul(self.f_drop, W)
      self.y_ = tf.mul(tf.cast(self.mask_y, tf.float32), tf.matmul(self.f_drop, W))

      self.probs = tf.nn.softmax(self.y_, name="probs")
      self.predictions = tf.argmax(self.y_, 1, name="predictions")

    with tf.variable_scope('loss'):
      losses = tf.nn.softmax_cross_entropy_with_logits(self.y_, self.input_y)
      self.loss = tf.reduce_mean(losses, name="loss")

    with tf.variable_scope('accuracy'):
      correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
      
      
  def get_input(self, data, train=True):
    '''
    convert data to tensorflow input
    data is a list of tuples (d,q,a)
    d, q, a is idx format
    !!!The data should be filter by max len!!!!!!
    '''
    xs = []
    ys = []
    ms = []
    for d,q,a in data:
      if self.qca:
        x = q + [data_utils.GO_ID] + d
      else:
        x = d + [data_utils.GO_ID] + q

      x_pad = x + [data_utils.PAD_ID] * (self.max_nsteps - len(x))
      xs.append(np.array(x_pad))

      y = np.zeros(self.vocab_size)
      y[np.array(a)] = 1
      ys.append(y)

      if train:
        m = np.ones(self.vocab_size)
      else:
        m = np.zeros(self.vocab_size)
        m[np.array(d)] = 1
      ms.append(m)   
    return xs, ys, ms


'''
test
'''

def main():
  import time
  data = [
    ([3,7,6,8,5,4,2], [3,4,5,6], [4]),
    ([2,3,4,5,6,7], [5,4,3], [7]),
    ([9,8,7,6,5,4,3], [3,5,8], [9])
  ]
  start = time.time()
  with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      model = DeepLSTM(size=4,
                      vocab_size=10,
                      depth=2,
                      batch_size=2,
                      max_nsteps=15)
      #sess.run(tf.global_variables_initializer())
      sess.run(tf.initialize_all_variables())
      batch_x, batch_y, batch_mask = model.get_input(data)
      dropout = 1.0
      feed_dict = {
        model.input_x:batch_x,
        model.input_y:batch_y,
        model.mask_y:batch_mask,
        model.dropout_keep_prob:dropout
      }
      
      print batch_x
      print batch_y
      print batch_mask

      '''
      #y_, feature, states = sess.run([model.y_, model.feature, model.states], feed_dict)
      #y_, loss, acc = sess.run([model.y_, model.loss, model.accuracy], feed_dict)
      print y_
      print loss
      print acc
      #print len(y_)
      '''
      '''
      print states
      print feature
      print y_
      '''
      
      print time.time() - start

if __name__ == '__main__':
  main()
