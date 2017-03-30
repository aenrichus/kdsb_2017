import tensorflow as tf
import pandas as pd
import numpy as np
import random
import time
import os

start_time = time.time()  # used to calculate elapsed time


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('pad_max', 400, 'Padding should be 400x400x400.')
flags.DEFINE_string('image_dir', 'stage1_processed/', 'Image directory')
flags.DEFINE_string('logs_dir', 'logs', 'Logs directory')

patients = os.listdir(FLAGS.image_dir)


def get_accuracy(predictions, labels):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1)), tf.float32))
    return accuracy


def run_model():
    # input placeholders
    x = tf.placeholder(tf.float32, [None, 128 * 32], name='x-input')
    x_image_shaped = tf.reshape(x, [-1, 400, 400, 400, 1])
    y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

    def weight_variable(shape):
        """
        Creates and initializes weights.

        :param shape: the number (shape) of weights you need
        :return: weights initialized on a truncated normal distribution
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # initialize biases
    def bias_variable(shape):
        """
        Creates and initializes biases.

        :param shape: the number (shape) of biases you need
        :return: biases initialized to a specific value (0.1)
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv_layer(input_tensor, input_kernel, input_dim, output_dim):
        """
        Makes a convolutional layer: It does a convolution, bias add, uses ReLU.

        :param input_tensor: the input to this layer
        :param input_kernel: number of dimensions (3)
        :param input_dim: number of feature maps in the input (1 for initial input)
        :param output_dim: number of feature maps in the output
        :return: output activations
        """
        weights = weight_variable([input_kernel, input_kernel, input_kernel, input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = tf.add(
            tf.nn.conv3d(input_tensor, weights, strides=[1, 1, 1, 1, 1], padding='SAME'), biases)
        activations = tf.nn.relu(preactivate)
        return activations

    def pool_layer(input_tensor, input_kernel = 2):
        pooled = tf.nn.max_pool3d(input_tensor,
                                  ksize=[1, input_kernel, input_kernel, input_kernel, 1],
                                  strides=[1, 1, 1, 1, 1],
                                  padding='SAME')
        return pooled

    def nn_layer(input_tensor, input_dim, output_dim, act=tf.nn.relu):
        """
        Makes a simple neural network layer: It does a matrix multiply, bias add, and then uses relu to nonlinearize.

        :param input_tensor: input to the layer
        :param input_dim: number of units in the input
        :param output_dim: number of units in the output
        :param act: activation function to use
        :return: output activations
        """
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = tf.add(tf.matmul(input_tensor, weights), biases)
        activations = act(preactivate)
        return activations

    # Model structure: AlexNet in 3D
    conv1 = conv_layer(x_image_shaped, 3, 1, 32)
    pool1 = pool_layer(conv1)
    conv2 = conv_layer(pool1, 3, 32, 64)
    conv3 = conv_layer(conv2, 3, 64, 128)
    conv4 = conv_layer(conv3, 3, 128, 256)
    conv5 = conv_layer(conv4, 3, 256, 512)
    pool2 = pool_layer(conv5)
    pool2_flat = tf.reshape(pool2, [-1, 100 * 100 * 100 * 512])
    fc1 = nn_layer(pool2_flat, 100 * 100 * 100 * 512, 4096, act=tf.nn.tanh)
    dropout1 = tf.nn.dropout(fc1, 0.5)
    fc2 = nn_layer(dropout1, 4096, 4096, act=tf.nn.tanh)
    dropout2 = tf.nn.dropout(fc2, 0.5)
    y = nn_layer(dropout2, 4096, 1, act=tf.nn.sigmoid)

    # Cost function: cross entropy between prediction and log output activations
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

    # Training using the Adam optimizer to minimize the overall cross entropy
    train_step = tf.train.MomentumOptimizer(0.0014, 0.9).minimize(cross_entropy)

    # TODO calculate accuracy of predictions
    accuracy = 0

    with tf.Session() as sess:

        print("Beginning training...")
        # Start tf session by initializing variables
        saver = tf.train.Saver()

        # TODO load the labels file
        # TODO divide into training and testing sets
        # TODO consider adding padding to the preprocessing
        # TODO check the maximum image size

        for i in range(FLAGS.max_steps+1):

            # select image
            current_image = np.random.choice(patients)

            # load the image
            current_subject = np.load(FLAGS.image_dir + current_image)

            # get the image dimensions
            current_shape = np.shape(current_subject)
            x_dim = current_shape[0]
            y_dim = current_shape[1]
            z_dim = current_shape[2]

            # determine dimensions for padding
            paddings = [[round((FLAGS.pad_max - x_dim)/2), round((FLAGS.pad_max - x_dim)/2)],
                        [round((FLAGS.pad_max - y_dim)/2), round((FLAGS.pad_max - y_dim)/2)],
                        [round((FLAGS.pad_max - z_dim)/2), round((FLAGS.pad_max - z_dim)/2)]]

            if x_dim % 2 == 1: paddings[0][0] -= 1
            if y_dim % 2 == 1: paddings[1][0] -= 1
            if z_dim % 2 == 1: paddings[2][0] -= 1

            # resize (pad) the image
            padded_subject = np.pad(current_subject, paddings)

            # TODO print test set cross entropy and accuracy
            if i % 100 == 0:
                ce, acc = sess.run([cross_entropy, accuracy], feed_dict={x: test_images, y_: test_labels})
                elapsed_time = round((time.time() - start_time) / 60, 2)
                print('STEP\t%s\tacc:\t%s\tce:\t%s\ttime\t%s minutes.' % (i, acc, ce, elapsed_time))

            # TODO train on padded image
            train_step.run(train_step, feed_dict={x: train_image, y_: train_label})

    print("Training complete.")

    # Save model state after training
    save_path = saver.save(sess, "logs/model_complete.ckpt")
    print("Model saved in file: %s" % save_path)


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    run_model()
    print("Game over?")

if __name__ == '__main__':
    tf.app.run()
