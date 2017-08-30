import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.ops import array_ops
from config import cnf

config_ = cnf.MedicalImages


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


class TSCGAN(object):
    def __init__(self):
        self.logits_ = None

    def G(self, training=True):
        with slim.arg_scope([slim.batch_norm], is_training=training):
            with slim.arg_scope([slim.convolution],
                                normalizer_fn=slim.batch_norm):
                with tf.variable_scope("G"):
                    noise = tf.random_normal(shape=[config_.BATCH_SIZE, config_.num_fc_1])
                    fc1 = slim.fully_connected(noise, num_outputs=config_.FEATURE_LEN,
                                               biases_initializer=PointInitializer(.1),
                                               weights_regularizer=slim.l2_regularizer(0.001), scope="fc1")
                    fc2 = slim.fully_connected(fc1, num_outputs=config_.FEATURE_LEN * config_.num_filt_2,
                                               biases_initializer=PointInitializer(.1),
                                               weights_regularizer=slim.l2_regularizer(0.001), scope="fc2")

                    fc2 = tf.reshape(fc2, [-1, config_.FEATURE_LEN, config_.num_filt_2])
                    deconv3 = slim.convolution(fc2, num_outputs=config_.num_filt_1, kernel_size=4,
                                               biases_initializer=PointInitializer(.1),
                                               weights_regularizer=slim.l2_regularizer(0.001), scope="deconv3")

                    deconv4 = slim.convolution(deconv3, num_outputs=1, kernel_size=5, activation_fn=None,
                                               biases_initializer=PointInitializer(.1),
                                               weights_regularizer=slim.l2_regularizer(0.001), scope="deconv4")
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
        with slim.arg_scope([slim.batch_norm], is_training=training, scale=True):
            with slim.arg_scope([slim.convolution],
                                normalizer_fn=slim.batch_norm):
                with tf.variable_scope("D", reuse=reuse):
                    conv1 = slim.convolution(inputs, num_outputs=config_.num_filt_1, kernel_size=9, stride=1,
                                             padding="SAME", biases_initializer=PointInitializer(scale=.1),
                                             weights_regularizer=slim.l2_regularizer(0.001), scope="Conv1")
                    # pool1 = slim.convolution(conv1, num_outputs=config_.num_filt_1, kernel_size=3, stride=2,
                    #                          padding='SAME', biases_initializer=PointInitializer(.1),
                    #                          weights_regularizer=slim.l2_regularizer(0.001), scope="Pool1")
                    pool1 = max_pool1d(conv1, kernel_size=2, stride=2, name="Pool1")
                    conv2 = slim.convolution(pool1, num_outputs=config_.num_filt_2, kernel_size=7, stride=1,
                                             padding="SAME", biases_initializer=PointInitializer(scale=.1),
                                             weights_regularizer=slim.l2_regularizer(0.001), scope="Conv2")
                    # pool2 = slim.convolution(conv2, num_outputs=config_.num_filt_2, kernel_size=3, stride=2,
                    #                          padding='SAME',biases_initializer=PointInitializer(.001),
                    #                          weights_regularizer=slim.l2_regularizer(.001), scope='Pool2')

                    pool2 = max_pool1d(conv2, kernel_size=2, stride=2, name="Pool2")

                    # flattened = tf.reshape(conv2, shape=[-1, np.prod(conv2.shape.as_list()[1:])], name="flatten")
                    # print(flattened)
                    # classification Branch
                    with tf.variable_scope("classification_branch"):
                        cls_conv3 = slim.convolution(pool2, num_outputs=config_.num_filt_3, kernel_size=7, stride=1,
                                                     padding='SAME', biases_initializer=PointInitializer(.1),
                                                     weights_regularizer=slim.l2_regularizer(0.001), scope='conv3')
                        cls_flattened = tf.reshape(cls_conv3, shape=[-1, np.prod(cls_conv3.shape.as_list()[1:])],
                                                   name="flatten")

                        cls_dropouted_flattened = tf.nn.dropout(cls_flattened, keep_prob=keep_prob)
                        fc1 = slim.fully_connected(cls_dropouted_flattened, num_outputs=config_.num_fc_1,
                                                   weights_regularizer=slim.l2_regularizer(0.001), scope="fc1")

                        #dropouted_fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
                        # [batch_size, 37]
                        logits = slim.fully_connected(fc1, num_outputs=config_.NUM_CLASSES,activation_fn=None,
                                                      weights_regularizer=slim.l2_regularizer(0.001), scope="logits")

                    # real fake branch
                    with tf.variable_scope("real_fake_branch"):
                        rf_conv3 = slim.convolution(pool2, num_outputs=config_.num_filt_3, kernel_size=7, stride=1,
                                                    padding='SAME', biases_initializer=PointInitializer(.1),
                                                    weights_regularizer=slim.l2_regularizer(0.001), scope='rf_conv3')
                        rf_flattened = tf.reshape(rf_conv3, shape=[-1, np.prod(rf_conv3.shape.as_list()[1:])],
                                                  name="flatten")
                        rf_dropouted_flattened = tf.nn.dropout(rf_flattened, keep_prob=keep_prob)
                        real_fake_fc1 = slim.fully_connected(rf_dropouted_flattened, num_outputs=config_.num_fc_1,

                                                             weights_regularizer=slim.l2_regularizer(0.001),
                                                             scope="fc1")
                        #dropouted_real_fake_fc1 = tf.nn.dropout(real_fake_fc1, keep_prob=keep_prob)
                        # [batch_size, 1]
                        real_fake_logits = slim.fully_connected(real_fake_fc1, num_outputs=1,
                                                                activation_fn=tf.sigmoid,
                                                                weights_regularizer=slim.l2_regularizer(0.001),
                                                                scope="logits")

        return logits, real_fake_logits

    def graph(self, inputs, labels):
        fake_data = self.G()
        # [batch_size, ]
        _, fake_prob = self.D(fake_data)
        logits, true_prob = self.D(inputs, reuse=True)

        self.logits_ = logits

        D_loss = -tf.reduce_mean(tf.log(true_prob) + tf.log(1 - fake_prob + 1e-8))
        G_loss = tf.reduce_mean(tf.log(1 - fake_prob + 1e-8), name="g_loss")
        classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        D_loss = tf.add(classification_loss, D_loss, name="d_loss")
        return D_loss, G_loss

    @staticmethod
    def accuracy(logits, labels, phrase='train'):
        """
        cal the accuracy
        :param logits: shape [batch_size, num_classes]
        :param labels: shape [batch_size, num_classes]
        :param phrase: 'train' or 'test'
        :return: the accuracy
        """
        if logits is None:
            raise ValueError("using self.logits_ before call graph() method")
        with tf.name_scope("Accuracy"):
            labels_id = tf.arg_max(labels, dimension=1, name="labels_id")

            print("label id ", labels_id)
            print("logits", logits)

            top1 = tf.nn.in_top_k(predictions=logits, targets=labels_id, k=1)
            top1 = tf.reduce_mean(tf.cast(top1, tf.float32), name=phrase + "_top1")
            top5 = tf.nn.in_top_k(predictions=logits, targets=labels_id, k=5)
            top5 = tf.reduce_mean(tf.cast(top5, tf.float32), name=phrase + "_top5")

            print(top1, top5)

        return top1, top5

    @staticmethod
    def get_opt(loss, D_or_G='D', regulizer=True):
        if regulizer:
            regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regu_loss = tf.reduce_mean(tf.convert_to_tensor(regu_loss))
            print(regu_loss)
        if D_or_G != 'D' and D_or_G != 'G':
            raise ValueError("D_or_G must be the string of 'D' or 'G'")

        # opt = tf.train.RMSPropOptimizer(0.001)
        opt = tf.train.AdamOptimizer(config_.LEARNING_RATE)
        loss = loss + regu_loss
        var_list = None
        if D_or_G == 'D':
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        else:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

        opt = tf.train.AdamOptimizer(config_.LEARNING_RATE)
        train_op = opt.minimize(loss, var_list=var_list)

        return train_op


def main():
    model = TSCGAN()
    inputs = tf.random_normal(shape=[config_.BATCH_SIZE, config_.FEATURE_LEN, 1])
    fake_data = model.D(inputs)
    print(model.G())
    print(fake_data)

    pass


if __name__ == '__main__':
    main()
