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

flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

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

  logger = open(os.path.join(FLAGS.model_dir, "dev_performance.log"), 'w')

  with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
      with tf.device('/gpu:%d' %(gpu)):
        model = model_dict[FLAGS.model](vocab_size = FLAGS.vocab_size, size=FLAGS.cell_size)
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate,
                                               decay=FLAGS.decay,
                                               momentum=FLAGS.momentum)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        grad_summaries = []
        for g, v in grads_and_vars:
          if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.join(FLAGS.model_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        print(" [+] Finish building model...")
        def train_step(batch_x, batch_y, batch_mask):
          """
          A single training step
          """
          feed_dict = {
            model.input_x: batch_x,
            model.input_y: batch_y,
            model.mask_y: batch_mask,
            model.dropout_keep_prob: FLAGS.dropout_keep_prob
          }
          _, step, summaries, loss, accuracy = sess.run(
              [train_op, global_step, train_summary_op, model.loss, model.accuracy],
              feed_dict)
          time_str = datetime.datetime.now().isoformat()
          print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
          train_summary_writer.add_summary(summaries, step)

        def dev_step(batch_x, batch_y, batch_mask, writer=None):
          """
          Evaluates model on a dev set
          """
          feed_dict = {
            model.input_x: batch_x,
            model.input_y: batch_y,
            model.mask_y: batch_mask,
            model.dropout_keep_prob: 1.0
          }
          step, summaries, loss, accuracy = sess.run(
              [global_step, dev_summary_op, model.loss, model.accuracy],
              feed_dict)
          time_str = datetime.datetime.now().isoformat()
          print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
          logger.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
          if writer:
            writer.add_summary(summaries, step)
        # load data
        print(" [*] Loading dataset...")
        train_data = data_utils.load_dataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.vocab_size, FLAGS.max_nsteps, part="training")
        dev_data = data_utils.load_dataset(FLAGS.data_dir, FLAGS.dataset, FLAGS.vocab_size, FLAGS.max_nsteps, part="validation")
        print(" [+] Finish loading. Train set: %d, Dev set: %d" %(len(train_data), len(dev_data)))
        # Generate batches
        dev_x, dev_y, dev_mask = model.get_input(dev_data, train=False)
        batches = data_utils.batch_iter(
            train_data, FLAGS.batch_size, FLAGS.epoch)
        # Training loop. For each batch...
        for batch in batches:
          batch_x, batch_y, batch_mask = model.get_input(batch)
          train_step(batch_x, batch_y, batch_mask)
          current_step = tf.train.global_step(sess, global_step)
          if current_step % FLAGS.evaluate_every == 0:
              print("\nEvaluation:")
              dev_step(dev_x, dev_y, dev_mask, writer=dev_summary_writer)
              print("")
          if current_step % FLAGS.checkpoint_every == 0:
              path = saver.save(sess, checkpoint_dir, global_step=current_step)
              print("Saved model checkpoint to {}\n".format(path))
  logger.close()
if __name__ == '__main__':
  tf.app.run()
