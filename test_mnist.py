
# coding: utf-8

# In[2]:

import os
import scipy.misc
import numpy as np

from utils import pp, visualize, to_json
from mnist import *

from sklearn.utils import shuffle

from random import randint

import tensorflow as tf


# In[3]:

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 2000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)


# In[4]:

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

sess = tf.InteractiveSession()

(x_train, y_train),(_,_),(x_test,y_test) = load_mnist()

train_size = x_train.shape[0]
test_size = x_test.shape[0]

epoch = 0
batch_count = 0

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits                               (labels=y_,logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(ITERS):
    # batch = mnist.train.next_batch(50)
    min_index = batch_count*BATCH_SIZE
    max_index = (batch_count+1)*BATCH_SIZE
    batch_count += 1
    
    if (max_index >= train_size):
        epoch += 1
        random_seed = randint(0,train_size)
        x_train, y_train = shuffle(x_train, y_train, random_state=random_seed)
        batch_count = 0 
        print("epoch %d, finished"%epoch)
        
    batch_images = x_train[min_index:max_index]
    batch_labels = y_train[min_index:max_index]
    
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_images, y_: batch_labels                                                  , keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: x_test, y_: y_test, keep_prob: 1.0}))


# # In[14]:

# x_train.shape


# # In[15]:

# y_train.shape


# # In[16]:

# random_seed = randint(0,train_size)


# # In[17]:

# random_seed


# # In[18]:

# x_train, y_train = shuffle(x_train, y_train, random_state=random_seed)


# # In[19]:

# x_train


# # In[ ]:



