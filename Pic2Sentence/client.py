import tensorflow as tf
import numpy as np
w = tf.get_variable("w",shape=[1,2], dtype=tf.float32)
x = tf.placeholder(shape=[None,1], dtype=tf.float32,name="x")
y = tf.placeholder(shape=[None,2], dtype=tf.float32, name="y")
y_ = tf.matmul(x,w)
init = tf.initialize_all_variables()
loss = tf.reduce_sum(tf.square(y-y_))
train_op = tf.train.GradientDescentOptimizer(0.0003).minimize(loss)
tf.scalar_summary("loss", loss)
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    writer = tf.train.SummaryWriter("/home/keith/workspace/tensorboard/",sess.graph)
    sess.run(init)
    for i in xrange(10000):
        x_v = [np.random.randn(1)]
        y_v = np.dot(x_v, [[2, 3]])
        feed_dict = {x: x_v, y: y_v}
        output_dict = [train_op, w]
        if i % 10 == 0:
            summary,_ = sess.run([merged, train_op],feed_dict)
            writer.add_summary(summary,i)
        else:
            sess.run(output_dict,feed_dict)
        print "step:%d" %(i)
