import nets.vgg_net as vgg
from tensorflow.contrib import slim
import tensorflow as tf
import time

bs = 110

inputs = tf.random_normal(shape=[bs, 224, 224, 3])
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
labels = tf.constant(value=[0]*bs, dtype=tf.int32)

with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)

train_op = tf.train.GradientDescentOptimizer(learning_rate=.0001).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    begin = time.time()
    for i in range(1000):
        print(i)
        sess.run(train_op)
    print("time ", time.time() - begin)


# batch size : 10,   8824M time:
# upper bound : 100
