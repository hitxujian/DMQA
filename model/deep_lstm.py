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


class DeepLSTM():
  """Deep LSTM model."""
  def __init__(self,
               size=256,
               vocab_size=100000,
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
    self.size = int(size)
    self.vocab_size = int(vocab_size)
    #self.batch_size = int(batch_size)
    self.max_nsteps = int(max_nsteps)
    self.qca = qca

    # Placeholders for input, output and dropout
    self.input_x = Input(shape=(self.max_nsteps,))

    print(" [*] Building Deep LSTM...")
    '''
    # embedding
    emb_input_x = Embedding(self.vocab_size, size, input_length=self.max_nsteps)(self.input_x)
    # deep LSTM
    x_zero = Lambda(lambda x: x*0.0)(emb_input_x)
    x1 = merge([emb_input_x, x_zero], mode='concat')
    h1 = LSTM(self.size, return_sequences=True)(x1)
    y1 = Dense(self.size)(h1)

    x2 = merge([emb_input_x, y1], mode='concat')
    h2 = LSTM(self.size)(x2)
    g = Dense(self.size)(h2)
    '''

    # embedding
    emb_input_x = Embedding(self.vocab_size, size, input_length=self.max_nsteps)(self.input_x)
    # deep LSTM
    h1 = LSTM(self.size, return_sequences=True)(emb_input_x)
    y1 = Dense(self.size)(h1)
    h2 = LSTM(self.size)(y1)
    g = Dense(self.size)(h2)
    self.y = Dense(self.vocab_size, activation='softmax')(g)

    self.model = Model(input=self.input_x, output=self.y)
    
    self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

  def train(self, train_set, dev_set, nb_epoch=1000, batch_size=3, model_dir=''):
    # prepare callbacks
    csv_logger = CSVLogger(os.path.join(model_dir, 'training.log'))
    checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir, "weights.hdf5"), verbose=1, save_best_only=True)
    tb = TensorBoard(log_dir=os.path.join(model_dir, 'logs'), histogram_freq=0, write_graph=True, write_images=False)
    # prepare data
    X, Y, _ = self.get_input(train_set)
    dev_X, dev_Y, _ = self.get_input(dev_set)
    self.model.fit(X, Y, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(dev_X, dev_Y), callbacks=[csv_logger, checkpointer, tb])


  def batch_train(self, train_set, dev_set, nb_epoch=1000, batch_size=3, model_dir='',
                  evaluate_every=100, checkpoint_every=1000):

    logger = open(os.path.join(model_dir, 'training.log'), 'w')
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    dev_X, dev_Y, _ = self.get_input(dev_set)
    step=0
    for train_data in data_utils.batch_iter(train_set, batch_size, nb_epoch):
      X, Y, _ = self.get_input(train_data)
      results = self.model.test_on_batch(X, Y)
      str_results = ', '.join(["%s: %.4f" %(k, v) for (k,v) in zip(self.model.metrics_names, results)])
      print("Step: %d, %s" %(step, str_results))
      logger.write("Step: %d, %s\n" %(step, str_results))
      self.model.train_on_batch(X,Y)
      step += 1
      if step % evaluate_every == 0:
        dev_results = self.model.test_on_batch(dev_X, dev_Y)
        str_dev_results = ', '.join(["%s: %.4f" %(k, v) for (k,v) in zip(self.model.metrics_names, dev_results)])
        print("Evaluate at dev set: %s" %(str_dev_results))
        logger.write("Evaluate at dev set: %s" %(str_dev_results))

      if step % checkpoint_every == 0:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint%d.hdf5" %(step))
        print("Save model to %s" %(checkpoint_path))
        self.model.save(checkpoint_path)
    logger.close()

      
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

      xs.append(x)

      y = np.zeros(self.vocab_size)
      y[np.array(a)] = 1
      ys.append(y)

      if train:
        m = np.ones(self.vocab_size)
      else:
        m = np.zeros(self.vocab_size)
        m[np.array(d)] = 1
      ms.append(m)
    pad_xs = pad_sequences(xs, maxlen=self.max_nsteps, padding='post')
    return pad_xs, np.array(ys), ms


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
  model = DeepLSTM(size=4,
                  vocab_size=10,
                  max_nsteps=15)

  xs, ys, ms = model.get_input(data)
  print(xs)
  print(ys)
  print(model.model.predict(xs))
  model.train(xs, ys, xs, ys)
  #model.model.fit(xs, ys, nb_epoch=1000, batch_size=3)
  model.batch_train(data, data, nb_epoch=600, batch_size=3, model_dir='toy_model')
  output = model.model.predict(xs)
  print(output)

if __name__ == '__main__':
  main()
