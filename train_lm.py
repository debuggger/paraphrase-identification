#! /usr/bin/env python
import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn_lm import TextCNNLM
from tensorflow.contrib import learn
from my_data_helper import *
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_data_file", "../data/corpus_max_tokens_truncated.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("test_data_file", "../data/msr_paraphrase_test.txt", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 1000, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
'''
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
'''

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    vocab = pickle.load(open('word_to_vec_max.dump'))
    embedding_size = np.shape(vocab[vocab.keys()[0]])[0]
    maxSentLen = getMaxSentLenLM(FLAGS.train_data_file)
    with sess.as_default():
        cnn = TextCNNLM(
            sequence_length=maxSentLen,
            num_classes=embedding_size,
            vocab_size=len(vocab),
            embedding_size=np.shape(vocab[vocab.keys()[0]])[0],
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        #acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        #train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        '''
        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
        '''

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model-lm")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, x1_batch, x2_batch, x3_batch, y_batch, epochNumber):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_x1: x1_batch,
              cnn.input_x2: x2_batch,
              cnn.input_x3: x3_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            #print np.shape(x_batch), np.shape(x1_batch), np.shape(x2_batch), np.shape(x3_batch), np.shape(y_batch)
            _, step, summaries, loss  = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: Epoch {} step {}, loss {:g}".format(time_str, epochNumber, step, loss))
            train_summary_writer.add_summary(summaries, step)
        '''
        def dev_step(x1_batch, x2_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x1: x1_batch,
              cnn.input_x2: x2_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
        '''

        # Generate batches
        #x1_dev, x2_dev, y_dev  = getNextBatch(FLAGS.test_data_file, vocab, 1, -1, maxSentLen)
       
        # Training loop. For each batch...
        for epochNumber in range(FLAGS.num_epochs):
            train_index = 1
            x_batch, x1_batch, x2_batch, x3_batch, y_batch = getNextBatchLM(FLAGS.train_data_file, vocab, train_index, FLAGS.batch_size)
 
            while len(x1_batch) > 0:
                train_step(x_batch, x1_batch, x2_batch, x3_batch, y_batch, epochNumber)
                current_step = tf.train.global_step(sess, global_step)
                '''
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x1_dev, x2_dev, y_dev, writer=dev_summary_writer)
                    print("")
                '''
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                train_index += FLAGS.batch_size
                x_batch, x1_batch, x2_batch, x3_batch, y_batch = getNextBatchLM(FLAGS.train_data_file, vocab, train_index, FLAGS.batch_size)
                

