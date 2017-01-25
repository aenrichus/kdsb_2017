import tensorflow as tf
import pandas as pd
import random
import time

start_time = time.time()  # used to calculate elapsed time


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_string('data_dir', 'data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', 'logs', 'Summaries directory')


def train_model():
    """
    This function contains the entire model. Inelegant, but it gets the job done.

    :return: None
    """
    print("Reading files...")
    train_data = pd.read_table("patterns/6ktraining.dict", header=None)
    test_data = pd.read_table("patterns/strain.txt", header=None)
    print("Read files.")

    print("Loading images...")
    train_images = load_images(train_data[0])
    test_images = load_images(test_data[0])
    print("Loaded", len(train_images) + len(test_images), "images.")

    print("Loading labels...")
    op_words = pd.read_table("patterns/op_words.txt", header=None)
    op_words_labs = op_char(op_words.ix[:, 2])
    op_train_labs = op_words_labs[:5870, :]  # np is not inclusive of final, right?
    op_test_labs = op_words_labs[5870:6030, :]
    print("Loaded", len(op_words_labs), "labels.")

    print("Instancing classes...")
    train = CnnData(train_images, op_train_labs)
    test = CnnData(test_images, op_test_labs)
    print("Instanced classes.")

    # Start the interactive session
    sess = tf.InteractiveSession()

    # input placeholders
    x = tf.placeholder(tf.float32, [None, 128 * 32], name='x-input')
    x_image_shaped = tf.reshape(x, [-1, 32, 128, 1])
    y_ = tf.placeholder(tf.float32, [None, 38 * 10], name='y-input')

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
        Makes a convolutional layer: It does a convolution, bias add, uses relu, and uses max-pooling.

        :param input_tensor: the input to this layer
        :param input_kernel: number of dimensions (3)
        :param input_dim: number of feature maps in the input (1 for initial input)
        :param output_dim: number of feature maps in the output
        :return: output activations
        """
        weights = weight_variable([input_kernel, input_kernel, input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME') + biases
        activations = tf.nn.relu(preactivate)
        pooled = tf.nn.max_pool(activations, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
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
        preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate)
        return activations

    # Model structure: input -> 3 convolutional layers => fully connected layer => output
    conv1 = conv_layer(x_image_shaped, 3, 1, 32)
    conv2 = conv_layer(conv1, 3, 32, 64)
    conv3 = conv_layer(conv2, 3, 64, 128)
    conv3_flat = tf.reshape(conv3, [-1, 16 * 4 * 128])
    fc = nn_layer(conv3_flat, 16*4*128, 500)
    y = nn_layer(fc, 500, 38*10, act=tf.nn.softmax)

    # Cost function: cross entropy between prediction and log output activations
    diff = y_ * tf.log(y)
    cross_entropy = -tf.reduce_sum(diff)

    # Training using the Adam optimizer to minimize the overall cross entropy
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

    # Evaluate the accuracy of predictions
    y0, y1, y2, y3, y4, y5, y6, y7, y8, y9 = tf.split(1, 10, y)  # reshape output to number of output locations
    y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9 = tf.split(1, 10, y_)

    correct_0 = tf.equal(tf.argmax(y0, 1), tf.argmax(y_0, 1))  # check if the most activated unit in y == y_
    correct_1 = tf.equal(tf.argmax(y1, 1), tf.argmax(y_1, 1))
    correct_2 = tf.equal(tf.argmax(y2, 1), tf.argmax(y_2, 1))
    correct_3 = tf.equal(tf.argmax(y3, 1), tf.argmax(y_3, 1))
    correct_4 = tf.equal(tf.argmax(y4, 1), tf.argmax(y_4, 1))
    correct_5 = tf.equal(tf.argmax(y5, 1), tf.argmax(y_5, 1))
    correct_6 = tf.equal(tf.argmax(y6, 1), tf.argmax(y_6, 1))
    correct_7 = tf.equal(tf.argmax(y7, 1), tf.argmax(y_7, 1))
    correct_8 = tf.equal(tf.argmax(y8, 1), tf.argmax(y_8, 1))
    correct_9 = tf.equal(tf.argmax(y9, 1), tf.argmax(y_9, 1))
    corr_01 = tf.logical_and(correct_0, correct_1)
    corr_23 = tf.logical_and(correct_2, correct_3)
    corr_45 = tf.logical_and(correct_4, correct_5)
    corr_67 = tf.logical_and(correct_6, correct_7)
    corr_89 = tf.logical_and(correct_8, correct_9)
    corr_0123 = tf.logical_and(corr_01, corr_23)
    corr_4567 = tf.logical_and(corr_45, corr_67)
    corr_0_7 = tf.logical_and(corr_0123, corr_4567)
    correct_prediction = tf.logical_and(corr_0_7, corr_89)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # convert to float and take the mean

    # Start tf session by initializing variables
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()

    print("Beginning training...")
    for i in range(FLAGS.max_steps+1):
        if i % 100 == 0:
            print('STEP\t%s\tElapsed time\t%s minutes.' % (i, round((time.time() - start_time) / 60, 2)))

            if i % 1000 == 0:  # Test on the training set
                ce, acc = sess.run([cross_entropy, accuracy], feed_dict={x: train.images, y_: train.labels})
                print('Train\tACC:\t%s\tERR:\t%s' % (acc, ce))

            # Test on Strain items (not in training set)
            ce, acc = sess.run([cross_entropy, accuracy], feed_dict={x: test.images, y_: test.labels})
            print('Test\tACC:\t%s\tERR:\t%s' % (acc, ce))

        else:  # Chose random input for training and run training
            batch_images, batch_labels = zip(*random.sample(list(zip(train.images, train.labels)), 10))
            sess.run(train_step, feed_dict={x: batch_images, y_: batch_labels})
    print("Training complete.")

    # Save model state after training
    save_path = saver.save(sess, "savegame/model_complete2.ckpt")
    print("Model saved in file: %s" % save_path)


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train_model()
    print("Game over?")

if __name__ == '__main__':
    tf.app.run()
