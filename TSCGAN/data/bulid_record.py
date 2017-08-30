import tensorflow as tf
import numpy as np
import os
from config.cnf import data_dir as DATADIR


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    try:
        len(value)
    except Exception:
        raise ValueError("value must be a list or ndarray")
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_to(data_set, name):
    """
    convert data set to tfrecords
    :param data_set: [num_sample, tuple(class, feature)] ziped label,feature
    :param name: test or train
    :return: Nothing
    """
    file_name = os.path.join("../data", name + ".tfrecords")
    print("writing tfrecords to ", file_name)
    writer = tf.python_io.TFRecordWriter(file_name)
    for lable, feature in data_set:
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": _int64_feature(lable),
            "feature": _float_feature(feature)
        }))
        writer.write(example.SerializeToString())

    writer.close()
    print("Done!!!!")


def build_dataset(file_name):
    data = np.loadtxt(file_name, delimiter=',')
    Labels = data[:, 0].astype(np.int64)

    # shift the label
    base = np.min(Labels)
    if base == -1:
        for i in range(len(Labels)):
            if Labels[i] == -1:
                Labels[i] = 0
    else:
        Labels -= base
    Feature = data[:, 1:]

    indices = np.arange(0, Feature.shape[0])
    np.random.shuffle(indices)

    feature = Feature[indices,]
    labels = Labels[indices,]

    num_cls = len(np.unique(labels))
    feature_len = feature.shape[1]
    dataset = zip(labels, feature)
    return dataset, num_cls, feature_len, Labels, Feature


def main():
    dataset_name = "Lighting2"
    data_dir = os.path.join(DATADIR, dataset_name)

    train_dataset_name = os.path.join(data_dir, dataset_name + "_TRAIN")

    train_dataset, num_cls, feature_len, labels, _ = build_dataset(train_dataset_name)

    convert_to(train_dataset, dataset_name + "_train")

    print("dataset " + dataset_name, "has %d classes" % num_cls, ", feature length is %d" % feature_len,
          ", num train data: ", len(labels))


if __name__ == '__main__':
    """
    Adiac
    num_cls = 37
    feature_len = 176
    
    MedicalImages
    num_cls = 10
    feature_len = 99
    
    Elec
    num_cls = 10
    feature_len = 96
    """
    main()
