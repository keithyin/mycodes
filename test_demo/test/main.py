import tensorflow as tf

val = tf.constant([1, 2, 3, 255], dtype=tf.uint8)

weight = tf.get_variable("weight", shape=[], dtype=tf.float32)

ass = tf.assign(weight, 3.0)

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print(sess.run(weight))
    saver.restore(sess, save_path="./model.ckpt")

    print(sess.run(weight))

    saver.save(sess, save_path="./model.ckpt")
