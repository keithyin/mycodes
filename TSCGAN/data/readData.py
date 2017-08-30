import tensorflow as tf

from config import cnf


def read_data(file_names, config, is_training=True):
    if isinstance(file_names, str):
        file_names = [file_names]
    assert isinstance(file_names, list)

    with tf.name_scope("ReadData"):
        file_name_queue = tf.train.string_input_producer(file_names, num_epochs=100000, shuffle=True)

        # prepare reader
        reader = tf.TFRecordReader()
        key, record_string = reader.read(file_name_queue)
        features = tf.parse_single_example(record_string, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'feature': tf.FixedLenFeature([config.FEATURE_LEN], tf.float32),
        })
        label = tf.cast(features['label'], dtype=tf.int32)
        feature = tf.expand_dims(features['feature'], axis=-1)

        min_after_dequeue = 200
        capacity = min_after_dequeue + 3 * config.BATCH_SIZE

        if is_training:
            example_batch, label_batch = tf.train.shuffle_batch([feature, label],
                                                                batch_size=config.BATCH_SIZE,
                                                                min_after_dequeue=0,
                                                                num_threads=4, capacity=config.num_train_data + 100)
        else:
            example_batch, label_batch = tf.train.shuffle_batch([feature, label],
                                                                batch_size=config.num_test_data,
                                                                min_after_dequeue=0,
                                                                num_threads=4, capacity=config.num_test_data + 100)

        # example_batch [batch_size, 176, 1], label_batch [batch_size]
        return example_batch, label_batch


def input_pipeline(train_tfrecords_names, test_tfrecords_names, config):
    """
    input pipeline
    :param train_tfrecords_names: the file names of training samples
    :param test_tfrecords_names: the file names of test samples
    :param batch_size: batch_size, int
    :return: example_batch, label_batch
    """
    with tf.name_scope("InputPipeLine"):
        train_exam_batch, train_label_batch = read_data(train_tfrecords_names, batch_size=config.BATCH_SIZE)
        test_exam_batch, test_label_batch = read_data(test_tfrecords_names, batch_size=config.num_test_data)

        for_training = tf.placeholder(tf.bool, name="for_train")  # feed it using for_train:0
        exam_batch, label_batch = tf.cond(for_training, true_fn=lambda: (train_exam_batch, train_label_batch),
                                          false_fn=lambda: (test_exam_batch, test_label_batch))

        # example_batch [batch_size, 176, 1], label_batch [batch_size]
        return exam_batch, label_batch


def main():
    file_name = "./Elec_test.tfrecords"
    conf = cnf.Elec()
    conf.BATCH_SIZE = 727
    example_batch, label_batch = read_data(file_names=file_name, config=conf)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        labels, features = sess.run([label_batch, example_batch])

        print(set(labels))
        # print(features)

        coord.request_stop()
        coord.join(threads)
        print("finished")
    pass


if __name__ == '__main__':
    main()
    pass
