import tensorflow as tf
from models import tsganv2
from config import cnf
import sys
import os
from tensorflow.contrib import slim

current_dir = sys.path[0]
config_ = cnf.Haptics

GLOBAL_STEP = 0
TRAIN_RECORDS = config_.TRAIN_RECORDS
TEST_RECORDS = config_.TEST_RECORDS
DataSetName = config_.DATASET_NAME

tf.set_random_seed(2017)
NUM_TRAIN_ITERATION = config_.num_train_data // config_.BATCH_SIZE


def net_suregery():
    train_batch_sample = tf.placeholder(dtype=tf.float32, shape=[15, 1092, 1])
    train_batch_label_one_hot = tf.placeholder(dtype=tf.float32, shape=[15, 5])

    # build TSCGAN

    model = tsganv2.TSCGAN(config=config_)
    d_loss, g_loss = model.graph(train_batch_sample, train_batch_label_one_hot)


def main():
    val = tf.placeholder(dtype=tf.float32, shape=[3, 100, 100, 3])
    training = tf.placeholder(dtype=tf.bool, shape=[])
    b = slim.batch_norm(val, is_training=training)
    res = tf.add(b, 1., name='add')
    gpu_conf = tf.GPUOptions(allow_growth=True)
    sess_conf = tf.ConfigProto(gpu_options=gpu_conf)
    with tf.Session(config=sess_conf) as sess:
        writer = tf.summary.FileWriter(logdir=os.path.join(current_dir, 'ckpt'), graph=sess.graph)


if __name__ == '__main__':
    main()
    pass
