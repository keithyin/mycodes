import sys
import tensorflow as tf
from PIL import Image
import progressbar
import os
import numpy as np
from skimage.transform import resize
import utils
import threading


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Sample(object):
    def __init__(self, img_path, label):
        self.img_path = img_path
        self.label = label


def convert_to(img_list, target_dir, target_height, target_width, name="train"):
    """
    conver raw image to tfrecords
    :param img_list:
    :param target_dir:
    :param target_height:
    :param target_width:
    :param name:
    :return:
    """

    if not isinstance(img_list, list):
        raise ValueError("img_list must be a list")

    total_num = len(img_list)

    ####
    widgets = ["processing: ", progressbar.Percentage(),
               " ", progressbar.ETA(),
               " ", progressbar.FileTransferSpeed(),
               ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=total_num).start()
    ####

    filename = os.path.join(target_dir, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for i, img in enumerate(img_list):
        bar.update(i)
        img_data = resize(np.array(Image.open(img.img_path)),
                          output_shape=(target_height, target_width)
                          , preserve_range=True).astype(np.uint8)
        depth = img_data.shape[2]
        img_data_raw = img_data.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(target_height),
            'width': _int64_feature(target_width),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(img.label)),
            'image_raw': _bytes_feature(img_data_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    bar.finish()
    print("done")


def main():
    file_name = \
        '/media/fanyang/workspace/DataSet/ai_challenger_scene_train_20170904/' \
        'scene_train_annotations_20170904.json'
    target_dir = '/media/fanyang/workspace/DataSet/SceneClassification'

    train_sample_list, val_sample_list = utils.tools.get_samples(file_name)
    t1 = threading.Thread(target=convert_to, args=(train_sample_list[:10000],
                                                   target_dir, 224, 224, 'train_0',))
    t2 = threading.Thread(target=convert_to, args=(train_sample_list[10000:20000],
                                                   target_dir, 224, 224, 'train_1',))
    t3 = threading.Thread(target=convert_to, args=(train_sample_list[20000:30000],
                                                   target_dir, 224, 224, 'train_2',))
    t4 = threading.Thread(target=convert_to, args=(train_sample_list[30000:40000],
                                                   target_dir, 224, 224, 'train_3',))
    t5 = threading.Thread(target=convert_to, args=(train_sample_list[40000:],
                                                   target_dir, 224, 224, 'train_4',))
    t6 = threading.Thread(target=convert_to, args=(val_sample_list,
                                                   target_dir, 224, 224, 'val_0',))

    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
if __name__ == '__main__':
    main()
