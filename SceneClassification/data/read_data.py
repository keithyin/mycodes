import tensorflow as tf
import configure
import matplotlib.pyplot as plt


def img_preprocess(inputs):
    outputs = tf.image.random_brightness(image=inputs, max_delta=.1)
    outputs = tf.image.random_flip_left_right(image=outputs)
    outputs = tf.image.random_flip_up_down(image=outputs)
    outputs = tf.image.random_contrast(image=outputs, lower=.3, upper=1.)
    return outputs


def read_data(file_names, height=224, width=224, batch_size=64, training=True):
    """

    :param file_names:
    :param height:
    :param width:
    :param batch_size:
    :return:
    """
    if isinstance(file_names, str):
        file_names = [file_names]
    with tf.name_scope("InputPipeLine"):
        file_name_queue = tf.train.string_input_producer(file_names, shuffle=True)

        # prepare reader
        reader = tf.TFRecordReader()
        key, record_string = reader.read(file_name_queue)
        features = tf.parse_single_example(record_string, features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)})
        # Note that choose the out_type carefully, must coincident with the dtype of data you encode
        raw_img = tf.decode_raw(features['image_raw'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        img_shape = [height, width, 3]
        img = tf.reshape(raw_img, shape=img_shape)
        img = tf.image.rgb_to_grayscale(images=img, name='rgb_2_gray_scale')
        if training:
            img = img_preprocess(img)
        img = tf.transpose(img, perm=[2, 0, 1])
        img = 2 * (1 - tf.cast(img, dtype=tf.float32) / 255.0)
        min_after_dequeue = 300
        capacity = min_after_dequeue + 3 * batch_size
        batch_img, batch_label = tf.train.shuffle_batch(tensors=[img, label],
                                                        batch_size=batch_size, capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue,
                                                        num_threads=4)
        return batch_img, batch_label


def read_data_demo(file_names, height=224, width=224, batch_size=64):
    """

    :param file_names:
    :param height:
    :param width:
    :param batch_size:
    :return:
    """
    if isinstance(file_names, str):
        file_names = [file_names]
    with tf.name_scope("InputPipeLine"):
        file_name_queue = tf.train.string_input_producer(file_names, shuffle=True)

        # prepare reader
        reader = tf.TFRecordReader()
        key, record_string = reader.read(file_name_queue)
        features = tf.parse_single_example(record_string, features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)})
        # Note that choose the out_type carefully, must coincident with the dtype of data you encode
        raw_img = tf.decode_raw(features['image_raw'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        img_shape = [height, width, 3]
        img = tf.reshape(raw_img, shape=img_shape)
        img = img_preprocess(img)
        return img


def main():
    dataset_cnf = configure.DataSetCnf()
    img = read_data_demo(dataset_cnf.train_records)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        img = sess.run([img])
        plt.imshow(img[0])
        plt.show()
        coord.request_stop()


if __name__ == '__main__':
    main()
