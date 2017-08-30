import tensorflow as tf
from data import readData
from data import bulid_record
from models import tsganv2
from config import cnf
import sys
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.python import debug as  tf_debug

current_dir = sys.path[0]
config_ = cnf.Beef

GLOBAL_STEP = 0
TRAIN_RECORDS = config_.TRAIN_RECORDS
TEST_RECORDS = config_.TEST_RECORDS
DataSetName = config_.DATASET_NAME

tf.set_random_seed(2017)
NUM_TRAIN_ITERATION = config_.num_train_data // config_.BATCH_SIZE


def train_num_iteration(sess, d_train_op, g_train_op, num_iteration):
    mean_top1 = 0.0
    mean_d_loss = 0.0
    for i in range(num_iteration):
        _, _, d_loss, train_top1 = sess.run([d_train_op, g_train_op, 'd_loss:0', 'Accuracy/train_top1:0'],
                                            feed_dict={'bn_train:0': True})
        mean_d_loss += d_loss
        mean_top1 += train_top1
    return mean_top1 / num_iteration, mean_d_loss / num_iteration


def test_num_iteration(sess, model, num_iteration=1):
    mean_top1 = 0.0
    threshold = 0.8
    for i in range(num_iteration):
        test_top1, true_label, pred = sess.run(['Accuracy_1/test_top1:0', "test_batch_label:0", "test_pred:0"],
                                               feed_dict={'bn_train:0': False})
        mean_top1 += test_top1

        print("ground truth: ", set(true_label))
        print("label preded: ", set(pred))
        print("num test data: ", len(pred))
        if test_top1 > .93:
            raise KeyboardInterrupt
        if test_top1 > threshold:
            # model.discount_d_loss_factor(discount_rate=0.)
            print(confusion_matrix(true_label, pred))
            threshold += .1
            if threshold > .9:
                raise KeyboardInterrupt
                # model.discount_d_loss_factor(discount_rate=0.) to discount d loss factor
                # model.set_discount_d_loss_factor(value=1.) to set the vale of d loss factor
    mean_top1 /= float(num_iteration)

    return mean_top1


def main():
    train_batch_sample, train_batch_label = readData.read_data(TRAIN_RECORDS, config=config_)
    # train_batch_sample = tf.add(train_batch_sample,
    #                             tf.random_normal(shape=tf.shape(train_batch_sample), stddev=0.1))
    ########################## prepare test data
    _, _, _, labels, features = bulid_record.build_dataset(config_.TEST_RECORDS)
    test_batch_sample = tf.Variable(features, trainable=False, dtype=tf.float32)
    test_batch_label = tf.Variable(labels, trainable=False, dtype=tf.int64)
    ############################################
    # test_batch_sample, test_batch_label = readData.read_data(TEST_RECORDS, config=config_, is_training=False)

    train_batch_label_one_hot = tf.one_hot(train_batch_label, depth=config_.NUM_CLASSES, name="train_one_hot")
    test_batch_label_one_hot = tf.one_hot(test_batch_label, depth=config_.NUM_CLASSES)

    # build TSCGAN

    model = tsganv2.TSCGAN(config=config_)
    d_loss, g_loss = model.graph(train_batch_sample, train_batch_label_one_hot)

    train_top1, train_top5 = model.accuracy(model.logits_, train_batch_label_one_hot)

    d_train_op = model.get_opt(d_loss, regulizer=False)
    g_train_op = model.get_opt(g_loss, D_or_G='G', regulizer=False)

    # g_train_op = model.get_opt(g_loss, D_or_G='G')

    # d for test
    test_logits, _ = model.D(test_batch_sample, reuse=True, training=False)

    ###########################################################################
    test_batch_label_ = tf.identity(test_batch_label, name="test_batch_label")
    test_pred = tf.arg_max(test_logits, dimension=1, name="test_pred")
    ###########################################################################

    test_top1, test_top5 = model.accuracy(test_logits, test_batch_label_one_hot, phrase='test')
    gpu_conf = tf.GPUOptions(allow_growth=True)
    sess_conf = tf.ConfigProto(gpu_options=gpu_conf)
    with tf_debug.LocalCLIDebugWrapperSession(tf.Session(config=sess_conf)) as sess:
        writer = tf.summary.FileWriter(logdir=os.path.join(current_dir, 'ckpt'), graph=sess.graph)
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_loss_track = []
        test_loss_track = []
        try:
            while not coord.should_stop():
                global GLOBAL_STEP
                # Run training steps or whatever
                train_accu, train_loss = train_num_iteration(sess=sess, d_train_op=d_train_op, g_train_op=g_train_op,
                                                             num_iteration=NUM_TRAIN_ITERATION)
                model.decay_learning_rate()

                test_accu = test_num_iteration(sess=sess, model=model, num_iteration=1)

                train_loss_track.append(train_accu)
                test_loss_track.append(test_accu)

                print("step %d, training loss %.6f , training accuracy %.6f , test accuracy %.6f." % (
                    GLOBAL_STEP * NUM_TRAIN_ITERATION, train_loss, train_accu, test_accu))
                if GLOBAL_STEP % 100 == 0:
                    model.save_ckpt(save_path=os.path.join(current_dir, "ckpt/GAN%s.ckpt" % DataSetName),
                                    global_step=GLOBAL_STEP)

                GLOBAL_STEP += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print('interrupted...')
            print('saving accu curve.')
            plt.plot(train_loss_track, label='train_accu')
            plt.plot(test_loss_track, label='test_accu')
            plt.legend(loc='lower right')
            plt.savefig('%s_gan_train_test.jpg' % config_.DATASET_NAME)
            print("max accu is ", max(test_loss_track))
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
