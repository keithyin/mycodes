import tensorflow as tf
from data import readData
from models import tsganv2
import sys
from config import cnf
import os
from data import bulid_record
from train_ganv2 import test_num_iteration

current_dir = sys.path[0]
config_ = cnf.Beef

GLOBAL_STEP = 0
TRAIN_RECORDS = config_.TRAIN_RECORDS
TEST_RECORDS = config_.TEST_RECORDS
DataSetName = config_.DATASET_NAME

tf.set_random_seed(2017)
NUM_TRAIN_ITERATION = 50
NUM_TEST_ITERATION = 50


def train_num_iteration(sess, d_train_op, num_iteration):
    mean_top1 = 0.0
    mean_d_loss = 0.0
    for i in range(num_iteration):
        _, d_loss, train_top1 = sess.run([d_train_op, 'd_loss:0', 'Accuracy/train_top1:0'],
                                         feed_dict={'bn_train:0': True})
        mean_d_loss += d_loss
        mean_top1 += train_top1
    return mean_top1 / num_iteration, mean_d_loss / num_iteration


def main():
    train_batch_sample, train_batch_label = readData.read_data(TRAIN_RECORDS, config=config_)
    # train_batch_sample = tf.add(train_batch_sample,
    #                             tf.random_normal(shape=tf.shape(train_batch_sample), stddev=0.1))
    train_batch_label_one_hot = tf.one_hot(train_batch_label, depth=config_.NUM_CLASSES, name="train_one_hot")
    ########################## prepare test data
    _, _, _, labels, features = bulid_record.build_dataset(config_.TEST_RECORDS)
    test_batch_sample = tf.Variable(features, trainable=False, dtype=tf.float32)
    test_batch_label = tf.Variable(labels, trainable=False, dtype=tf.int64)
    test_batch_label_one_hot = tf.one_hot(test_batch_label, depth=config_.NUM_CLASSES)
    ############################################

    # build TSCGAN
    model = tsganv2.TSCGAN(config=config_)
    logits, _ = model.D(train_batch_sample)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_batch_label_one_hot, logits=logits)
    loss = tf.reduce_mean(loss, name='d_loss')
    train_top1, train_top5 = model.accuracy(logits, train_batch_label_one_hot)

    d_train_op = model.get_opt(loss, regulizer=False)
    # g_train_op = model.get_opt(g_loss, D_or_G='G')

    # d for test
    test_logits, _ = model.D(test_batch_sample, reuse=True, training=False)

    ###########################################################################
    test_batch_label_ = tf.identity(test_batch_label, name="test_batch_label")
    test_pred = tf.arg_max(test_logits, dimension=1, name="test_pred")
    ###########################################################################

    test_top1, test_top5 = model.accuracy(test_logits, test_batch_label_one_hot, phrase='test')

    d_var_list = tf.get_collection(
        tf.GraphKeys.SAVEABLE_OBJECTS, scope="D") + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D')

    d_saver = tf.train.Saver(var_list=d_var_list)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=os.path.join(current_dir, 'ckpt_cnn'), graph=sess.graph)
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                global GLOBAL_STEP
                # Run training steps or whatever
                train_accu, train_loss = train_num_iteration(sess=sess, d_train_op=d_train_op,
                                                             num_iteration=NUM_TRAIN_ITERATION)
                model.decay_learning_rate()

                test_accu = test_num_iteration(sess=sess, model=model, num_iteration=NUM_TEST_ITERATION)
                print("step %d, training loss %.6f , training accuracy %.6f , test accuracy %.6f." % (
                    GLOBAL_STEP * NUM_TRAIN_ITERATION, train_loss, train_accu, test_accu))

                d_saver.save(sess=sess, save_path=os.path.join(current_dir, "ckpt_cnn/%s.ckpt" % DataSetName),
                             global_step=GLOBAL_STEP)
                GLOBAL_STEP += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print('interrupted...')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            # Wait for threads to finish.
        coord.join(threads)
        sess.close()
        print("all threads are closed! ")


if __name__ == '__main__':
    main()
    pass
