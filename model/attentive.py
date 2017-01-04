# coding=utf-8
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, merge, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
from keras import backend as K

import numpy as np
seed = 7
np.random.seed(seed)

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
import data_utils



def reverse_seq(X):
  return X[:,::-1,:]


class Attentive():
  """Deep LSTM model."""
  def __init__(self,
               size=256,
               vocab_size=100000,
               max_dsteps=1000,
               max_qsteps=100):
    """Initialize the parameters for an Attentive model.
    
    Args:
      size: int, The dimensionality of the inputs into the Deep LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      keep_prob: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      max_nsteps: int, The max time unit [1000]
      qca: if true: the input is q+d, else: the input is d+q
    """
    self.size = int(size)
    self.vocab_size = int(vocab_size)
    #self.batch_size = int(batch_size)
    self.max_dsteps = int(max_dsteps)
    self.max_qsteps = int(max_qsteps)

    # Placeholders for input, output and dropout
    self.input_d = Input(shape=(self.max_dsteps,))
    self.input_q = Input(shape=(self.max_qsteps,))

    print(" [*] Building Attentive Model...")
    
    # embedding
    Embedding_layer = Embedding(self.vocab_size, size)
    emb_input_d = Embedding_layer(self.input_d)
    emb_input_q = Embedding_layer(self.input_q)

    lstm = LSTM(self.size, return_sequences=True, go_backwards=True)
    lstm_q = lstm(emb_input_q)
    lstm_q = Lambda(reverse_seq)(lstm_q)

    # Bi LSTM
    
    lstm_fwd_d = LSTM(self.size, return_sequences=True, name='lstm_fwd_d')(emb_input_d)
    lstm_bwd_d = LSTM(self.size, return_sequences=True, go_backwards=True, name='lstm_bwd_d')(emb_input_d)
    lstm_bwd_d = Lambda(reverse_seq)(lstm_bwd_d)
    bi_lstm_d = merge([lstm_fwd_d, lstm_bwd_d], name='bilstm_d', mode='concat')

    lstm_fwd_q = LSTM(self.size, name='lstm_fwd_q')(emb_input_q)
    lstm_bwd_q = LSTM(self.size, go_backwards=True, name='lstm_bwd_q')(emb_input_q)
    u = merge([lstm_fwd_q, lstm_bwd_q], name='bilstm_q', mode='concat')

    # Attention
    
    
    self.model = Model(input=[self.input_d, self.input_q], output=bi_lstm_d)
    
    self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

  def train(self, X, Y, dev_X, dev_Y, nb_epoch=1000, batch_size=3):
    csv_logger = CSVLogger('training.log')
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)
    self.model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(dev_X, dev_Y), callbacks=[csv_logger, checkpointer, tb])
      
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
      ds.append(d)
      qs.append(q)

      y = np.zeros(self.vocab_size)
      y[np.array(a)] = 1
      ys.append(y)

      if train:
        m = np.ones(self.vocab_size)
      else:
        m = np.zeros(self.vocab_size)
        m[np.array(d)] = 1
      ms.append(m)
    pad_ds = pad_sequences(ds, maxlen=self.max_dsteps, padding='post')
    pad_qs = pad_sequences(qs, maxlen=self.max_qsteps, padding='post')
    return pad_ds, pad_qs, np.array(ys), ms


'''
test
'''

def main():
  import time
  data = [
    ([3,7,6,8,5,4,2], [3,4,5], [4]),
    ([2,3,4,5,6,7], [5,4,3], [7]),
    ([9,8,7,6,5,4,3], [3,5,8], [9])
  ]
  model = Attentive(size=4,
                  vocab_size=10,
                  max_dsteps=10,
                  max_qsteps=3)

  ds, qs, ys, ms = model.get_input(data)
  print(ds)
  print(qs)
  print(ys)
  print(model.model.predict([ds, qs]))
  #model.train(xs, ys, xs, ys)
  #model.model.fit(xs, ys, nb_epoch=1000, batch_size=3)
  #output = model.model.predict(xs)
  #print(output)

if __name__ == '__main__':
  main()
