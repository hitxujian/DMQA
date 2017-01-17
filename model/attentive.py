# coding=utf-8
from keras.models import Model
from keras.layers import *
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

def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.batch_dot(Y, K.expand_dims(alpha, dim=-1))
    return ans


class AttentiveReader():
  """Deep LSTM model."""
  def __init__(self,
               size=256,
               vocab_size=100000,
               max_dsteps=1000,
               max_qsteps=100):
    """Initialize the parameters for an Attentive model.
    
    Args:
      size: int, The dimensionality of the inputs into the LSTM cell [32, 64, 256]
      learning_rate: float, [1e-3, 5e-4, 1e-4, 5e-5]
      batch_size: int, The size of a batch [16, 32]
      keep_prob: unit Tensor or float between 0 and 1 [0.0, 0.1, 0.2]
      max_dsteps: int, The max time unit of document [1000]
      max_qsteps: int, The max time unit of query [1000]
    """
    self.size = int(size)
    self.hidden_size = 2 * self.size
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
    y = merge([lstm_fwd_d, lstm_bwd_d], name='bilstm_d', mode='concat')

    lstm_fwd_q = LSTM(self.size, name='lstm_fwd_q')(emb_input_q)
    lstm_bwd_q = LSTM(self.size, go_backwards=True, name='lstm_bwd_q')(emb_input_q)
    u = merge([lstm_fwd_q, lstm_bwd_q], name='bilstm_q', mode='concat')

    # Attention
    Wum = Dense(self.hidden_size, bias=False, name="Wum")(u)
    Wum = RepeatVector(self.max_dsteps, name="Wum_n")(Wum)
    Wym = TimeDistributed(Dense(self.hidden_size, bias=False), name="Wym")(y)
    m_ = merge([Wum, Wym], mode='sum')
    m = Activation('tanh', name='m')(m_)
    alpha_ = TimeDistributed(Dense(1), name="alpha_")(m)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Activation('softmax', name='alpha')(flat_alpha)
    #alpha = K.expand_dims(alpha, dim=-1)
    y_trans = Permute((2, 1), name="y_trans")(y)

    r_ = merge([y_trans, alpha], output_shape=(self.hidden_size, 1), name="r_", mode=get_R)
    r = Flatten(name="flat_r")(r_)
 
    Wug = Dense(self.hidden_size, bias=False, name="Wug")(u)
    Wrg = Dense(self.hidden_size, bias=False, name="Wrg")(r)
    g_ = merge([Wrg, Wug], mode='sum')
    g = Activation('tanh', name='g')(g_)

    self.y = Dense(self.vocab_size, activation='softmax')(g)
    
    self.model = Model(input=[self.input_d, self.input_q], output=self.y)
    
    self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

  def train(self, train_set, dev_set, nb_epoch=1000, batch_size=3):
    csv_logger = CSVLogger('training.log')
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False)

    D,Q,Y,_ = self.get_input(train_set)
    dD,dQ,dY,_ = self.get_input(dev_set)
    self.model.fit([D,Q], Y, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([dD,dQ], dY), callbacks=[csv_logger, checkpointer, tb])
      
  
  def batch_train(self, train_set, dev_set, nb_epoch=1000, batch_size=3, model_dir='',
                  evaluate_every=100, checkpoint_every=1000):

    logger = open(os.path.join(model_dir, 'training.log'), 'w')
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    dD,dQ,dY,_ = self.get_input(dev_set)
    step=1
    for train_data in data_utils.batch_iter(train_set, batch_size, nb_epoch):
      D,Q,Y,_ = self.get_input(train_data)
      results = self.model.test_on_batch([D, Q], Y)
      str_results = ', '.join(["%s: %.4f" %(k, v) for (k,v) in zip(self.model.metrics_names, results)])
      print("Step: %d, %s" %(step, str_results))
      logger.write("Step: %d, %s\n" %(step, str_results))
      self.model.train_on_batch([D, Q],Y)
      
      if step % evaluate_every == 0:
        dev_results = self.model.test_on_batch([dD,dQ], dY)
        str_dev_results = ', '.join(["%s: %.4f" %(k, v) for (k,v) in zip(self.model.metrics_names, dev_results)])
        print("Evaluate at dev set: %s" %(str_dev_results))
        logger.write("Evaluate at dev set: %s\n" %(str_dev_results))

      if step % checkpoint_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint%d.hdf5" %(step))
        print("Save model to %s" %(checkpoint_path))
        self.model.save(checkpoint_path)

      step += 1
    logger.close()

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
  model.train(data, data)
  #model.model.fit(xs, ys, nb_epoch=1000, batch_size=3)
  #output = model.model.predict(xs)
  #print(output)

if __name__ == '__main__':
  main()
