import tensorflow as tf
import gradient

zero_out = tf.load_op_library("./zero_out.so").zero_out
w = tf.constant([1,2,3])
val = zero_out(w)

res = val[0] + val[1]

grad = tf.gradients(res, w)

with tf.Session() as sess:
    print(sess.run(grad))