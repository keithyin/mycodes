import tensorflow as tf
import matplotlib.pyplot as plt

def read_data(file_names, batch_size):
    if isinstance(file_names, str):
        file_names = [file_names]
    assert isinstance(file_names, list)

    with tf.name_scope("InputPipeLine"):
        file_name_queue = tf.train.string_input_producer(file_names, num_epochs=10000, shuffle=True)

        # prepare reader
        reader = tf.TFRecordReader()
        key, record_string = reader.read(file_name_queue)
        features = tf.parse_single_example(record_string,features={
                                           'height': tf.FixedLenFeature([],tf.int64),
                                            'width': tf.FixedLenFeature([], tf.int64),
                                            'depth': tf.FixedLenFeature([], tf.int64),
                                            'label': tf.FixedLenFeature([], tf.int64),
                                            'image_raw': tf.FixedLenFeature([],tf.string)})
        img = tf.decode_raw(features['image_raw'], tf.float32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        label = tf.cast(features['label'], tf.int32)
        img_shape = tf.stack([height, width, depth])
        img_reshaped = tf.cast(tf.reshape(img,[224,224,3]), tf.float32)
        # if the shape parameter in tf.reshape is tensor, there will be an error in 
        # tf.train.shuffle_batch. don't know why, probably must have the same shape
        # within one batch

        min_after_dequeue = 500
        capacity = min_after_dequeue+3*batch_size

        example_batch, label_batch = tf.train.shuffle_batch([img_reshaped, label],
                                                            batch_size=batch_size,min_after_dequeue=min_after_dequeue,
                                                            num_threads=4, capacity=capacity)
        return example_batch, label_batch


def main():
    exam_bat,label_bat = read_data("/media/dafu/262C99212C98ECD5/tfrecords/train.tfrecords", 32)
    #img, label = read_data("/media/dafu/262C99212C98ECD5/tfrecords/train.tfrecords", 32)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        labels, imgs = sess.run([label_bat, exam_bat])
        print(labels)
        print(imgs.dtype)
        print(imgs[0][0])
        fig = plt.figure()
        for i in range(32):
            axis = fig.add_subplot(6,6, i+1)
            axis.imshow(imgs[i])
        plt.show()
        coord.request_stop()



if __name__ == '__main__':
    main()

