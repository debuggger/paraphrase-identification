from preprocess import *
import tensorflow as tf

def cnnTrain(word_embeddings, vec_size, h=3):
	def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)

	def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)

	def conv2d(x, W, vec_size):
	  return tf.nn.conv2d(x, W, strides=[1, vec_size, 1, 1], padding='SAME')

	def avg_pool_2x1(x):
	  return tf.nn.avg_pool(x, ksize=[1, 2, 1, 1],
				strides=[1, 2, 1, 1], padding='SAME')
	
	x = tf.placeholder("float", shape=[None, shape[1]*shape[0]])
	y_ = tf.placeholder("float", shape=[None, 2])


	W_conv1 = weight_variable([vec_size, 3, 1, 6])
	b_conv1 = bias_variable([6])

	x = tf.reshape(x, [-1,shape[1],shape[0],1])

	h_conv1 = conv2d(x, W_conv1)
	h_pool1 = tf.nn.relu(avg_pool_2x1(h_conv1) + b_conv1)

	W_conv2 = weight_variable([vec_size/2, 5, 6, 12])
	b_conv2 = bias_variable([12])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = avg_pool_2x1(h_conv2)
	'''	
	W_conv3 = weight_variable([5, 5, 32, 64])
	b_conv3 = bias_variable([64])

	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)

	W_fc1 = weight_variable([3 * 16 * 64, 512])
	b_fc1 = bias_variable([512])

	h_pool3_flat = tf.reshape(h_pool3, [-1, 3*16*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
	'''	
	W_fc1 = weight_variable([6 * 32 * 32, 768])
	b_fc1 = bias_variable([768])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 6*32*32])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([768,2])
	b_fc2 = bias_variable([2])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		batch_size = 1000
		for i in range(10):
			start = 0
			print 'i=',i
			print 'Num batches:', int(len(train_x)/batch_size)
			nb = int(len(train_x)/batch_size)
			for j in range(nb): 
				batch_x = train_x[start: start+batch_size]
				batch_y = train_y[start: start+batch_size]
				
				train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
				print("step %d,batch (%d/%d), training accuracy %g"%(i,j,nb,train_accuracy))
				train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
				start += batch_size
			print("test accuracy %g"%accuracy.eval(feed_dict={x: train_x[:2000], y_: train_y[:2000], keep_prob: 1.0}))


		saver.save(sess, 'model_cnn_2_6.ckpt')
		print("test accuracy %g"%accuracy.eval(feed_dict={
		    x: train_x[:1000], y_: train_y[:1000], keep_prob: 1.0}))

if __name__ == '__main__':
	cnnTrain(word_embeddings, vec_size, h)
