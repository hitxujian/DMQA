import tensorflow as tf
import numpy as np
from base_model import Model

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import data_utils

class AttentiveReader(Model):
  """Attentive Reader."""
  def __init__(self,
               size=256,
               vocab_size=100000,
               max_nsteps=1000,
               max_nquery=100):
    """Initialize the parameters for an  Attentive Reader model.

    Args:
      vocab_size: int, The dimensionality of the input vocab
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      att_size: The size of attention [100]
      max_nsteps: int, The max nsteps of document [1000]
      max_nquery: int, The max nsteps of query [100]

    The attentive Reader:
    u = state of bidirection_rnn(q)   [batch_size * (2*size)]
    
    # y is not the output is an annotation from the paper
    y = output of bidirection_rnn(d)  [max_nsteps * batch_size * (2*size)]

    m = tanh(W_ym * y + W_um * u)     [max_nsteps * batch_size * size]

    s = transpose(softmax(W_ms * m))  [batch_size * max_nsteps]

    r = y * s                         [batch_size * (2*size)]

    g = tanh(W_rg * r + W_ug * u)     [batch_size * feature_size]

    output = W_y * g                       [batch_size * vocab_size]

    """
    super(AttentiveReader, self).__init__()

    self.vocab_size = vocab_size
    self.size = size
    self.max_nsteps = int(max_nsteps)
    self.max_nquery = max_nquery

    # Placeholders for input, output and dropout
    self.input_d = tf.placeholder(tf.int32, [None, self.max_nsteps], name="input_x")
    self.input_q = tf.placeholder(tf.int32, [None, self.max_nsteps], name="input_x")
    self.input_y = tf.placeholder(tf.int32, [None, self.vocab_size], name="input_y")
    self.mask_y = tf.placeholder(tf.int32, [None, self.vocab_size], name="mask_y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    self.emb = tf.get_variable("emb", [self.vocab_size, self.size])
    embed_input_d = tf.nn.embedding_lookup(self.emb, tf.transpose(self.input_d))
    embed_input_q = tf.nn.embedding_lookup(self.emb, tf.transpose(self.input_q))

    batch_size = tf.shape(self.input_d)[0]


    # build bidirection lstm for q
    with tf.variable_scope('q_bid_LSTM'):
      q_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
      q_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
      _, self.q_states_fw, self.q_states_bw = tf.nn.bidirectional_rnn(q_cell_fw,
                                             q_cell_bw,
                                             tf.unpack(embed_input_q),
                                             dtype=tf.float32)
      self.u = tf.concat(1, [self.q_states_fw.h, self.q_states_bw.h])

    # build bidirection lstm for d
    with tf.variable_scope('d_bid_LSTM'):
      d_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
      d_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.size)
      self.y, _, _ = tf.nn.bidirectional_rnn(d_cell_fw,
                                                                      d_cell_bw,
                                                                      tf.unpack(embed_input_d),
                                                                      dtype=tf.float32)
    with tf.variable_scope('attention'):
      m = []
      W_ym = tf.get_variable(
          "W_ym",
          shape=[2*self.size, self.size],
          initializer=tf.contrib.layers.xavier_initializer()) 

      W_um = tf.get_variable(
          "W_um",
          shape=[2*self.size, self.size],
          initializer=tf.contrib.layers.xavier_initializer()) 

      for y_t in self.y:
        m.append(tf.tanh(tf.matmul(y_t, W_ym) + tf.matmul(self.u, W_um)))
      self.m = m

      w_ms = tf.get_variable(
        "w_ms",
          shape=[self.size, 1],
          initializer=tf.contrib.layers.xavier_initializer())
      s = []
      for m_t in self.m:
        s.append(tf.matmul(m_t, w_ms))

      self.rs = tf.reshape(s, [self.max_nsteps, -1])
      self.s = tf.nn.softmax(tf.transpose(self.rs))

      #self.r = 





  def get_input(self, data, train=True):
    '''
    convert data to tensorflow input
    data is a list of tuples (d,q,a)
    d, q, a is idx format
    !!!The data should be filter by max len!!!!!!
    '''
    ds = []
    qs = []
    ys = []
    ms = []
    for d,q,a in data:
      d_pad = d + [data_utils.PAD_ID] * (self.max_nsteps - len(d))
      ds.append(np.array(d_pad))

      q_pad = q + [data_utils.PAD_ID] * (self.max_nquery - len(q))
      qs.append(np.array(d_pad))

      y = np.zeros(self.vocab_size)
      y[np.array(a)] = 1
      ys.append(y)

      if train:
        m = np.ones(self.vocab_size)
      else:
        m = np.zeros(self.vocab_size)
        m[np.array(d)] = 1
      ms.append(m)   
    return ds, qs, ys, ms


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
      model = AttentiveReader(size=4,
                      vocab_size=10,
                      max_nsteps=15,
                      max_nquery=5)
      #sess.run(tf.global_variables_initializer())
      sess.run(tf.initialize_all_variables())
      batch_d, batch_q, batch_y, batch_mask = model.get_input(data)
      dropout = 1.0
      feed_dict = {
        model.input_d:batch_d,
        model.input_q:batch_q,
        model.input_y:batch_y,
        model.mask_y:batch_mask,
        model.dropout_keep_prob:dropout
      }
      
      print 'input'
      print batch_d
      print batch_q
      print batch_y
      print batch_mask

      
      u, y, s, rs = sess.run([model.u, model.y, model.s, model.rs], feed_dict)
      print 'output'
      print u
      print y
      print s
      print rs

      
      print time.time() - start

if __name__ == '__main__':
  main()


    


