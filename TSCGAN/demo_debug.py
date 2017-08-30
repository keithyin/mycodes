import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

IMAGE_SIZE = 28
HIDDEN_SIZE = 500
NUM_LABELS = 10
RAND_SEED = 42


def main(_):
    # Input placeholders.
    with tf.name_scope("input"):
        x = tf.random_normal(shape=[32, 28 * 28])

        y_ = tf.random_normal(shape=[32, 10], name="y-input")

    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1, seed=RAND_SEED)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer."""
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope("weights"):
                weights = weight_variable([input_dim, output_dim])
            with tf.name_scope("biases"):
                biases = bias_variable([output_dim])
            with tf.name_scope("Wx_plus_b"):
                preactivate = tf.matmul(input_tensor, weights) + biases

            activations = act(preactivate)
            return activations

    hidden = nn_layer(x, IMAGE_SIZE ** 2, HIDDEN_SIZE, "hidden")
    logits = nn_layer(hidden, HIDDEN_SIZE, NUM_LABELS, "output", tf.identity)
    y = tf.nn.softmax(logits)

    with tf.name_scope("cross_entropy"):
        diff = -(y_ * tf.log(y))
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(diff)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if FLAGS.debug:
        print("for debug")
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Add this point, sess is a debug wrapper around the actual Session if
    # FLAGS.debug is true. In that case, calling run() will launch the CLI.
    for i in range(FLAGS.max_steps):
        acc = sess.run(accuracy)
        print("Accuracy at step %d: %s" % (i, acc))

        sess.run(train_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Number of steps to run trainer.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=100,
        help="Batch size used during training.")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.025,
        help="Initial learning rate.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp/mnist_data",
        help="Directory for storing data")
    parser.add_argument(
        "--ui_type",
        type=str,
        default="curses",
        help="Command-line user interface type (curses | readline)")
    parser.add_argument(
        "--fake_data",
        type="bool",
        nargs="?",
        const=True,
        default=False,
        help="Use fake MNIST data for unit testing")
    parser.add_argument(
        "--debug",
        type="bool",
        nargs="?",
        const=True,
        default=True,
        help="Use debugger to track down bad values during training")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
