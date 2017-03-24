# import tensorflow as tf
# import os
# w = tf.get_variable("w",shape=[1])
#
# op = tf.assign(w, [6])
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#
#     #sess.run(op)
#     if os.path.exists("test/model.ckpt"):
#         saver.restore(sess,"test/model.ckpt")
#     print(sess.run(w))
#     saver.save(sess,"test/model.ckpt")

import os
if not os.path.exists("checkpoint_rnn"):
    os.mkdir("checkpoint_rnn")
    print("hello")