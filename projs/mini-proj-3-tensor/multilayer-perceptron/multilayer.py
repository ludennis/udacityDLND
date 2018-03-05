import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def batchize(features,labels,batch_size):
	assert (len(features)==len(labels))
	batches = []
	for i in range(0,len(features),batch_size):
		if i+batch_size < len(features):
			batches.append([features[i:i+batch_size],labels[i:i+batch_size]])
		else:
			batches.append([features[i:len(features)],labels[i:len(features)]])
	return np.array(batches).T[0], np.array(batches).T[1]

if __name__ == '__main__':
	mnist_data = input_data.read_data_sets('MNIST_data',one_hot=True)

	# set hypermeters
	n_epoch = 100
	learning_rate = 0.001
	batch_size = 128
	n_hidden_layer = 32

	# save file storing weights and biases variables
	save_file = './model.ckpt'

	# loading tensorflow's mnist dataset
	train_features = mnist_data.train.images 						#(55000,784)
	train_labels = mnist_data.train.labels.astype(np.float32) 		#(55000,10)

	valid_features = mnist_data.validation.images					#(5000,784)
	valid_labels = mnist_data.validation.labels.astype(np.float32)	#(5000,10)
	
	test_features = mnist_data.test.images							#(10000,784)
	test_labels = mnist_data.test.labels.astype(np.float32)			#(10000,10)

	n_features = train_features.shape[1]
	n_labels = train_labels.shape[1]

	# split data in batches
	train_feature_batches,train_label_batches = batchize(train_features,train_labels,batch_size)
	valid_feature_batches,valid_label_batches = batchize(valid_features,valid_labels,batch_size)
	test_feature_batches,test_label_batches = batchize(test_features,test_labels,batch_size)

	# define layers by creating tensors
	# input (batch_size, 784)
	x = tf.placeholder(shape=(None,n_features),dtype=tf.float32)
	y = tf.placeholder(shape=(None,n_labels),dtype=tf.float32)
	output = tf.placeholder(shape=(batch_size,n_labels),dtype=tf.float32)
	keep_prob = tf.placeholder(tf.float32)


	weights = {'hidden_layer':tf.Variable(tf.truncated_normal(shape=(n_features,n_hidden_layer),name='weights_0')),
			   'output':tf.Variable(tf.truncated_normal(shape=(n_hidden_layer,n_labels),name='weights_1'))}
	biases = {'hidden_layer':tf.Variable(tf.random_normal(shape=[n_hidden_layer],name='biases_0')),
			  'output':tf.Variable(tf.random_normal(shape=[n_labels],name='biases_1'))}

	# hidden layer
	hidden_layer = tf.add(tf.matmul(x,weights['hidden_layer']),biases['hidden_layer'])
	hidden_layer_output = tf.nn.relu(hidden_layer)
	hidden_layer_output = tf.nn.dropout(hidden_layer_output,keep_prob)

	# output layer
	logits = tf.add(tf.matmul(hidden_layer_output,weights['output']),biases['output'])

	# optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

	# calculate accuracy
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(n_epoch):
			for train_feature_batch,train_label_batch in zip(train_feature_batches,train_label_batches):
				result = sess.run(optimizer,feed_dict={x:train_feature_batch,y:train_label_batch,keep_prob:0.5})
			if epoch % 10 == 0 :
				valid_accuracy = sess.run(accuracy,feed_dict={x:valid_features,y:valid_labels,keep_prob:1.0})
				print('Epoch {} - Validation Accuracy: {}'.format(epoch,valid_accuracy))
		tf.train.Saver().save(sess,save_file)
		print ('Trained Model Saved.')

	# with tf.Session() as sess:
	# 	tf.train.Saver().restore(sess,save_file)
	# 	test_accuracy = sess.run(accuracy,feed_dict={x:test_features,y:test_labels})

	# print ('Test Accuracy: {}'.format(test_accuracy))
