import tensorflow as tf

queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string, tf.int64])

en_many = queue.enqueue_many([['hello', 'world', 'gg'], [3, 4, 6]])
en = queue.enqueue(['hello', 3])
de = queue.dequeue()

# queue = tf.FIFOQueue(capacity=10, dtypes=tf.string)
#
# en = queue.enqueue_many([['hello', 'good', 'gg'], ])
# # en = queue.enqueue(['hello', 3])
# de = queue.dequeue()

with tf.Session() as sess:
    sess.run(en)
    print(sess.run(de))
    print(sess.run(de))
    print(sess.run(de))
    print(sess.run(de))
    print(sess.run(de))
    print(sess.run(de))
