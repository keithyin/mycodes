import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.training import queue_runner
from data.generate_record import parse_txt_file


def img_preprocess(image):
    """

    :param image: uint8
    :return: uint8
    """
    image = tf.image.resize_bilinear(images=[image], size=(299, 299))
    image = tf.squeeze(image, axis=0)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image


def read_data(file_names, batch_size, training=True):
    """
    Using to read a mini-batch from the tfrecords.
    :param file_names: specify the tfrecords file names
    :param batch_size: batch size
    :return: a tuple of (example_batch, label_batch)
    """

    if isinstance(file_names, str):
        file_names = [file_names]
    assert isinstance(file_names, list)

    with tf.name_scope("InputPipeLine"):
        file_name_queue = tf.train.string_input_producer(file_names, num_epochs=10000, shuffle=True)

        # prepare reader
        reader = tf.TFRecordReader()
        key, record_string = reader.read(file_name_queue)
        features = tf.parse_single_example(record_string, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)})
        # Note that choose the out_type carefully, must coincident with the dtype of data you encode
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)

        # if you want to use shuffle_batch, you must specicy the img shape
        # img_reshaped = tf.cast(tf.shape(img, img_shape), tf.float32) doesn't work for shuffle batch
        img_reshaped = tf.reshape(img, [256, 128, 3])

        if training:
            img_reshaped = img_preprocess(img_reshaped)

        img_reshaped = tf.cast(img_reshaped, tf.float32)
        img_reshaped = img_reshaped / 255.0 - tf.constant([0.485, 0.456, 0.406])
        img_reshaped = img_reshaped / tf.constant([0.229, 0.224, 0.225])
        img_reshaped = tf.transpose(img_reshaped, perm=[2, 0, 1])

        min_after_dequeue = 500
        capacity = min_after_dequeue + 3 * batch_size

        example_batch, label_batch = tf.train.shuffle_batch([img_reshaped, label],
                                                            batch_size=batch_size, min_after_dequeue=min_after_dequeue,
                                                            num_threads=4, capacity=capacity)
        return example_batch, label_batch, None


def main():
    train_records = ["/media/fanyang/workspace/DataSet/MARS/train_%d.tfrecords" % i for i in range(45)]
    test_records = "/media/fanyang/workspace/DataSet/MARS/test.tfrecords"
    _, _, img = read_data(train_records, batch_size=10)

    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        iii = sess.run(img)
        plt.imshow(iii)
        plt.show()
        coord.request_stop()

    pass


class DatasetTF(object):
    def __init__(self, root, txt_file):
        self.data = parse_txt_file(txt_file, data_dir=root)
        self.flag = True

    def get_dataset(self):
        if not self.flag:
            raise ValueError("Can't call this method twice")
        self.flag = False
        strings = []
        ints = []
        for obj in self.data:
            strings.append(obj.img_path)
            ints.append(obj.label)

        q = string_int_pair_producer(strings_ints=[strings, ints])
        return q


def string_int_pair_producer(strings_ints):
    capacity = len(strings_ints[0])

    q = data_flow_ops.RandomShuffleQueue(capacity=capacity,
                                         dtypes=[tf.string, tf.int64], min_after_dequeue=0,
                                         name="name_label_queue")
    enq = q.enqueue_many(strings_ints)
    queue_runner.add_queue_runner(
        queue_runner.QueueRunner(q, [enq]))

    return q


class CustomDataLoaderTF(object):
    def __init__(self, string_int_pair_queue, batch_size, shape=(299, 299)):
        self.sess = tf.get_default_session()
        if self.sess is None:
            raise ValueError('this method must be running within with tf.Session block')

        self.queue = string_int_pair_queue
        self.img_shape = shape
        self.batch_size = batch_size

        self.deque_op = self.queue.dequeue()
        self.batch_data = None
        self.batch_label = None
        self.threads = None
        self.coord = None

        self.__fill_data_queue()
        self.__start_queue_runners()

    def __next__(self):
        return self.sess.run([self.batch_data, self.batch_label])

    def __fill_data_queue(self):
        file_name, label = self.deque_op
        # file_name.set_shape([])

        # Have been blocked here, The Reader need the file_name queue
        reader = tf.WholeFileReader()
        key, file = reader.read(queue=self.queue)

        # file = tf.gfile.GFile(
        #     name=file_name, mode='rb').read()
        img = tf.image.decode_jpeg(file, channels=3)
        img = self.transform(img)

        label.set_shape([])

        min_after_deque = 100
        capacity = min_after_deque + 3 * self.batch_size
        batch_data, batch_label = tf.train.shuffle_batch(tensors=[img, label],
                                                         batch_size=self.batch_size,
                                                         capacity=capacity,
                                                         min_after_dequeue=min_after_deque,
                                                         num_threads=4, name="data_queue")
        self.batch_data = batch_data
        self.batch_label = batch_label

    def __start_queue_runners(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def join(self):
        self.coord.join(self.threads)

    def transform(self, img):
        img = tf.image.resize_bilinear(images=[img], size=self.img_shape)
        img = tf.squeeze(img, axis=0)
        return img

    def __iter__(self):
        return self


if __name__ == '__main__':
    dataset = DatasetTF(root="/media/fanyang/workspace/DataSet/MARS/bbox_train",
                        txt_file="../data/test.txt")
    q = dataset.get_dataset()
    # de = q.dequeue()
    # file = tf.gfile.FastGFile(
    #     name='/media/fanyang/workspace/DataSet/MARS/bbox_train/0273/0273C2T0003F211.jpg', mode='rb').read()
    # img = tf.image.decode_jpeg(file, channels=3)
    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        # tf.train.start_queue_runners(sess=sess, coord=coord)
        # print(sess.run(de))
        # plt.imshow(sess.run(img))
        # plt.show()
        loader = CustomDataLoaderTF(string_int_pair_queue=q, batch_size=1)

        for data, label in loader:
            plt.imshow(data[0])
            plt.show()
            exit()
        loader.join()
