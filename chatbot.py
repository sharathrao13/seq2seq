from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import preprocessing_data as prepros
# Use tensorflows translate seq2seq model
from tensorflow.models.rnn.translate import seq2seq_model

vocab_path = './vocabulary_for_movies.txt'

# Variables user can change
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model")
tf.app.flags.DEFINE_integer("vocab_size", 7000, "Vocabulary size")
tf.app.flags.DEFINE_boolean("use_lstm", False, "Use LSTM as cell")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding")

FLAGS = tf.app.flags.FLAGS


# Static variables
learning_rate = 0.5
learning_rate_decay = 0.99
train_dir = "./"
steps_per_checkpoint = 50
gradients_clip = 5.0
num_movie_scripts = 2318


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path):
 
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target:

        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(prepros.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size, FLAGS.vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, gradients_clip, FLAGS.batch_size,
      learning_rate, learning_rate_decay, use_lstm = FLAGS.use_lstm,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    #prepros.make_files(num_movie_scripts,FLAGS.vocab_size)
    session.run(tf.initialize_all_variables())
  return model


def train():

  print ("Training the model")

  en_train = './X_train.txt'
  fr_train = './y_train.txt'
  en_dev   = './y_dev.txt'
  fr_dev   = './X_dev.txt'

  with tf.Session() as sess:
    # Create model.
    print("Creating " + str(FLAGS.num_layers) + " layers of " + str(FLAGS.size))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data")
    dev_set = read_data(en_dev, fr_dev)
    train_set = read_data(en_train, fr_train)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:

      ran_num = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale)) if train_buckets_scale[i] > ran_num])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / steps_per_checkpoint
      loss += step_loss / steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        if loss < 300:
          perplexity = math.exp(loss)
        else:
          perplexity = float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(train_dir, "chatbot.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
          if eval_loss < 300:
            eval_ppx = math.exp(eval_loss)
          else:
            eval_ppx = float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    vocab, rev_vocab = prepros.initialize_vocabulary(vocab_path)

    # Decode from standard input.
    sys.stdout.write("Human >: ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = prepros.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
      # Which bucket does it belong to?
      # Find the smallest bucket that fits
      bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch( {bucket_id: [(token_ids, [])]}, bucket_id )
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if prepros.EOS_ID in outputs:
        outputs = outputs[:outputs.index(prepros.EOS_ID)]

      print("Ola >: " + " ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
      print("Human >: ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()