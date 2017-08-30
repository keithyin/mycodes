import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops import array_ops


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, kernel=1):
    if isinstance(kernel, int):
        kernel = [kernel] * 2
    return tf.nn.conv2d(x, W, strides=[1, *kernel, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


initializer = tf.contrib.layers.xavier_initializer()


def max_pool1d(inputs, kernel_size=2, stride=2, name=None):
    with tf.name_scope(name, default_name="Pool1D"):
        assert len(inputs.get_shape().as_list()) == 3
        expanded_inputs = tf.expand_dims(inputs, axis=1)  # [batch_size, 1, feature_len, channel]
        pooled = slim.max_pool2d(expanded_inputs, kernel_size=(1, kernel_size), stride=(1, stride), padding="SAME")
    squeezed_pooled = tf.squeeze(pooled, axis=1)
    assert len(squeezed_pooled.get_shape().as_list()) == 3
    assert squeezed_pooled.get_shape().as_list()[-1] == inputs.get_shape().as_list()[-1]
    return squeezed_pooled


################################


class PointInitializer(Initializer):
    def __init__(self, scale, dtype=tf.float32):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return self.scale * array_ops.ones(shape, dtype)


class GAN(object):
    def __init__(self, config):
        self.logits_ = None
        self.bn_train = tf.placeholder(dtype=tf.bool, name='bn_train')
        self.config = config
        self.learning_rate_decay = None
        self.saver_for_GAN = None
        self.saver_for_D = 'Not None'
        self.fake_data = None
        self.D_loss_factor = None
        self.D_loss_discounted_factor = tf.placeholder(dtype=tf.float32, name='d_loss_discounted_factor')
        self.D_loss_factor_ph = tf.placeholder(dtype=tf.float32, name='d_loss_factor_ph')

    def G(self, training=True):
        with slim.arg_scope([slim.batch_norm], is_training=training, scale=True):
            with slim.arg_scope([slim.convolution],
                                normalizer_fn=slim.batch_norm):
                with tf.variable_scope("G"):
                    noise = tf.random_normal(shape=[self.config.BATCH_SIZE, self.config.num_fc_1])
                    fc1 = slim.fully_connected(noise, num_outputs=self.config.FEATURE_LEN,
                                               biases_initializer=PointInitializer(.1),
                                               scope="fc1")
                    fc2 = slim.fully_connected(fc1, num_outputs=self.config.FEATURE_LEN * self.config.num_filt_2,
                                               biases_initializer=PointInitializer(.1),
                                               scope="fc2")

                    fc2 = tf.reshape(fc2, [-1, self.config.FEATURE_LEN, self.config.num_filt_2])
                    deconv3 = slim.convolution(fc2, num_outputs=self.config.num_filt_1, kernel_size=4,
                                               biases_initializer=PointInitializer(.1),
                                               scope="deconv3")

                    deconv4 = slim.convolution(deconv3, num_outputs=1, kernel_size=5, activation_fn=None,
                                               biases_initializer=PointInitializer(.1),
                                               scope="deconv4", normalizer_fn=None)
                    # deconv4 = tf.nn.relu(deconv4)
            return deconv4

    def D(self, inputs, reuse=False, training=True):
        """
        feed forward procedure
        :param inputs:  shape [batch_size, time_step, channel]
        :return:
        """
        keep_prob = 1.0
        if training:
            keep_prob = 0.5
        norm_scale = False
        with tf.variable_scope("D", reuse=reuse):
            with tf.name_scope("Reshaping_data") as scope:
                x_image = tf.reshape(inputs, [-1, self.config.FEATURE_LEN, 1, 1])
            with tf.name_scope("Conv1") as scope:
                a_conv1 = slim.conv2d(x_image, num_outputs=self.config.num_filt_1, kernel_size=[5, 1], scope='conv1')

            with tf.name_scope('Batch_norm_conv1') as scope:
                a_conv1 = slim.batch_norm(a_conv1, is_training=self.bn_train, scale=norm_scale,
                                          updates_collections=None)
                h_conv1 = tf.nn.relu(a_conv1)

                # h_conv1 = slim.avg_pool2d(h_conv1, kernel_size=2, stride=2, padding='SAME')

            with tf.variable_scope('real_fake_branch'):
                with tf.name_scope("Conv2") as scope:
                    W_conv2 = tf.get_variable("Conv_Layer_2",
                                              shape=[4, 1, self.config.num_filt_1, self.config.num_filt_2],
                                              initializer=initializer)
                    b_conv2 = bias_variable([self.config.num_filt_2], 'bias_for_Conv_Layer_2')
                    a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

                with tf.name_scope('Batch_norm_conv2') as scope:
                    a_conv2 = slim.batch_norm(a_conv2, is_training=self.bn_train, updates_collections=None)
                    h_conv2 = tf.nn.relu(a_conv2)

                with tf.name_scope("Fully_Connected1") as scope:
                    W_fc1 = tf.get_variable("Fully_Connected_layer_1",
                                            shape=[self.config.FEATURE_LEN * self.config.num_filt_2,
                                                   self.config.num_fc_1],
                                            initializer=initializer)
                    b_fc1 = bias_variable([self.config.num_fc_1], 'bias_for_Fully_Connected_Layer_1')
                    h_conv3_flat = tf.reshape(h_conv2, [-1, self.config.FEATURE_LEN * self.config.num_filt_2])
                    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

                with tf.name_scope("Fully_Connected2"):
                    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
                    W_fc2 = tf.get_variable("W_fc2", shape=[self.config.num_fc_1, 1],
                                            initializer=initializer)
                    b_fc2 = tf.Variable(tf.constant(0.1, shape=[1]), name='b_fc2')
                    real_fake_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                    real_fake_logits = tf.sigmoid(real_fake_logits)

            return real_fake_logits

    def graph(self, inputs):

        self.fake_data = self.G()
        # [batch_size, ]
        fake_prob = self.D(self.fake_data)
        true_prob = self.D(inputs, reuse=True)

        D_loss = tf.negative(tf.reduce_mean(tf.log(true_prob) + tf.log(1 - fake_prob)), name='d_loss')
        G_loss = tf.reduce_mean(tf.log(1 - fake_prob + 1e-8), name="g_loss")
        return D_loss, G_loss

    def get_opt(self, loss, D_or_G='D', regulizer=True):
        regu_loss = 0.
        if regulizer:
            var_list = tf.trainable_variables()
            regulizer = slim.l2_regularizer(0.0005)
            regu_loss = slim.apply_regularization(regulizer, weights_list=var_list)

        if D_or_G != 'D' and D_or_G != 'G':
            raise ValueError("D_or_G must be the string of 'D' or 'G'")

        # opt = tf.train.RMSPropOptimizer(0.001)
        initial_learning_rate = tf.Variable(self.config.LEARNING_RATE, trainable=False, dtype=tf.float32)
        self.learning_rate_decay = tf.assign(initial_learning_rate, initial_learning_rate * .99)

        opt = tf.train.AdamOptimizer(initial_learning_rate)
        loss = loss + regu_loss
        var_list = None
        if D_or_G == 'D':
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        else:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

        opt = tf.train.AdamOptimizer(self.config.LEARNING_RATE)
        train_op = opt.minimize(loss, var_list=var_list)

        return train_op

    def decay_learning_rate(self):
        sess = tf.get_default_session()
        if sess is None or self.learning_rate_decay is None:
            raise ValueError('need session learning rate decay op')
        sess.run(self.learning_rate_decay)

    def _prepare_saver(self):
        """
        using this method to prepare saver
        :return: None
        """
        if self.saver_for_D is None:
            d_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
            self.saver_for_D = tf.train.Saver(var_list=d_var_list)
        if self.saver_for_GAN is None:
            gan_var_list = tf.trainable_variables()
            self.saver_for_GAN = tf.train.Saver(var_list=gan_var_list)

    def save_ckpt(self, save_path, global_step, model='GAN'):
        self._prepare_saver()
        if model != 'GAN' and model != 'D':
            raise ValueError('model is the string, GAN , D')

        if model == 'GAN':
            print('saving ckpt')
            self.saver_for_GAN.save(sess=tf.get_default_session(), save_path=save_path,
                                    global_step=global_step)
        else:
            print('saving ckpt')
            self.saver_for_D.save(sess=tf.get_default_session(), save_path=save_path,
                                  global_step=global_step)

    def restore_from_latest_ckpt(self, ckpt_dir, model='GAN'):
        self._prepare_saver()
        if model != 'GAN' and model != 'D':
            raise ValueError('model is the string, GAN or D')
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if model == 'GAN':
                self.saver_for_GAN.restore(tf.get_default_session(), ckpt.model_checkpoint_path)
            else:
                self.saver_for_D.restore(tf.get_default_session(), ckpt.model_checkpoint_path)
        print('restore has done, ready to inference...')


def main():
    # model = TSCGAN()
    # inputs = tf.random_normal(shape=[config.BATCH_SIZE, self.config.FEATURE_LEN, 1])
    # fake_data = model.D(inputs)
    # print(model.G())
    # print(fake_data)
    pass


if __name__ == '__main__':
    main()
