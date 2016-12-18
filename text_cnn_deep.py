import tensorflow as tf
import numpy as np


class TextCNNDeep(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters_1, num_filters_2, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        '''
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        '''
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_1a = []
        pooled_outputs_1b = []
        pooled_outputs_2a = []
        pooled_outputs_2b = []
        self.x1 = tf.expand_dims(self.input_x1, -1)
        self.x2 = tf.expand_dims(self.input_x2, -1)
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-1a-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters_1]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_1]), name="b")
                conv = tf.nn.conv2d(
                    self.x1,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, 2, 1],
                    strides=[1, 1, 2, 1],
                    padding='SAME',
                    name="pool")
                pooled_outputs_1a.append(pooled)
            
            with tf.name_scope("conv-maxpool-2a-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters_1]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters_1]), name="b")
                    conv = tf.nn.conv2d(
                        self.x2,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1],
                        padding='SAME',
                        name="pool")
                    pooled_outputs_2a.append(pooled)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-1b-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size/2, num_filters_1*len(filter_sizes), num_filters_2]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_2]), name="b")
                conv = tf.nn.conv2d(
                    tf.reshape(tf.concat(3, pooled_outputs_1a), [-1, sequence_length, embedding_size/2, num_filters_1*len(filter_sizes)]),
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_1b.append(pooled)
            
            with tf.name_scope("conv-maxpool-2b-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size/2, num_filters_1*len(filter_sizes), num_filters_2]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters_2]), name="b")
                    conv = tf.nn.conv2d(
                        tf.reshape(tf.concat(3, pooled_outputs_2a), [-1, sequence_length, embedding_size/2, num_filters_1*len(filter_sizes)]),
                        W,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs_2b.append(pooled)

        print '1A->', map(lambda x: x.get_shape(), pooled_outputs_1a)
        print '2A->', map(lambda x: x.get_shape(), pooled_outputs_2a)
        print '1B->', map(lambda x: x.get_shape(), pooled_outputs_1b)
        print '2B->', map(lambda x: x.get_shape(), pooled_outputs_2b)

        # Combine all the pooled features
        num_filters_total = 2* num_filters_2 * len(filter_sizes)
        print 'num_filters_total->',num_filters_total
        self.h_pool = tf.concat(3, pooled_outputs_1b + pooled_outputs_2b)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, (embedding_size/2) * num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total * (embedding_size/2), num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        self._1a_shape = tf.shape(pooled_outputs_1a[0])
        self._1b_shape = tf.shape(pooled_outputs_1b[0])
        self._score_shape = tf.shape(self.scores)
        self._h_pool_shape = tf.shape(self.h_pool)
        self._h_pool_flat_shape = tf.shape(self.h_pool_flat)
        print 'shapes:', num_filters_total, self.h_pool.get_shape(), self.h_pool_flat.get_shape(), self.h_drop.get_shape(), self.scores.get_shape(), self.input_y.get_shape()
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
