import os
import scipy.misc
import numpy as np

import time

from utils import pp, visualize, to_json
from mnist import *
from usps import *
from svhn import *

import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

flags = tf.app.flags

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 20000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1/10
# config.gpu_options.allow_growth = True

# with tf.Session(config=config) as sess:

session = tf.InteractiveSession()

# dataset
# (x_train, y_train),(x_valid, y_valid),(x_test, y_test) = load_mnist()
(x_train, y_train),(x_valid, y_valid) = load_usps()

train_size = x_train.shape[0]
valid_size = x_valid.shape[0]

real_x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_x = Generator(BATCH_SIZE)

disc_real = Discriminator(real_x)
disc_fake = Discriminator(fake_x)

# y_ = tf.placeholder(tf.float32, shape=[None, 10])

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

alpha = tf.random_uniform(
    shape=[BATCH_SIZE,1], 
    minval=0.,
    maxval=1.
)

differences = fake_x - real_x
interpolates = real_x + (alpha*differences)

gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5,
    beta2=0.9
).minimize(gen_cost, var_list=gen_params)

disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4, 
    beta1=0.5, 
    beta2=0.9
).minimize(disc_cost, var_list=disc_params)

clip_disc_weights = None

fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samp = Generator(128, noise=fixed_noise)

def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samp)
    lib.save_images.save_images(samples.reshape((128, 28, 28)), 
                                'samples_{}.png'.format(frame))

session.run(tf.initialize_all_variables())

epoch = 0

for iteration in range(ITERS):
    start_time = time.time()

    if iteration > 0:
        _ = session.run(gen_train_op)

    batch_count = 0

    for i in xrange(CRITIC_ITERS):

        min_index = batch_count*BATCH_SIZE
        max_index = (batch_count+1)*BATCH_SIZE
        
        batch_count += 1
        
        if (max_index > train_size):
            epoch += 1
            random_seed = randint(0,train_size)
            x_train, y_train = shuffle(x_train, y_train, random_state=random_seed)
            batch_count = 0 
            
            min_index = batch_count*BATCH_SIZE
            max_index = (batch_count+1)*BATCH_SIZE

            batch_count += 1

            print("epoch %d, finished"%epoch)

        batch_images = x_train[min_index:max_index]
        batch_labels = y_train[min_index:max_index]

        _data = batch_images
        _disc_cost, _ = session.run([disc_cost, disc_train_op],
                                    feed_dict={real_x: _data})

    lib.plot.plot('train disc cost', _disc_cost)
    lib.plot.plot('time', time.time() - start_time)

    if iteration % 100 == 99:
        valid_disc_costs = []
        valid_count = 0

        while valid_count < (valid_size // BATCH_SIZE):
            
            valid_min_index = valid_count*BATCH_SIZE
            valid_max_index = (valid_count+1)*BATCH_SIZE

            if (valid_max_index > valid_size):
                random_seed = randint(0,valid_size)
                x_valid, y_valid = shuffle(x_valid, y_valid, random_state=random_seed)
                valid_count = 0 
                break

            valid_images = x_valid[valid_min_index:valid_max_index]
            valid_labels = y_valid[valid_min_index:valid_max_index]

            valid_count += 1

            _valid_disc_cost = session.run(
                disc_cost, 
                feed_dict={real_x: valid_images}
            )
            valid_disc_costs.append(_valid_disc_cost)

        lib.plot.plot('dev disc cost', np.mean(valid_disc_costs))

        generate_image(iteration, _data)

    # Write logs every 100 iters
    if (iteration < 5) or (iteration % 100 == 99):
        lib.plot.flush()

    lib.plot.tick()