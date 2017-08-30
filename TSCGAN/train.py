import tensorflow as tf
from data import readData
from models import basic


GLOBAL_STEP = 0
TRAIN_RECORDS = "data/MedicalImages_train.tfrecords"
TEST_RECORDS = "data/MedicalImages_test.tfrecords"
DataSetName = "MedicalImages"
tf.set_random_seed(2017)


def train_num_iteration(sess, d_train_op, g_train_op, num_iteration=200):
    print("training.........................global step ", GLOBAL_STEP)
    mean_top1 = 0.0
    mean_g_loss = 0.0
    mean_d_loss = 0.0
    for i in range(num_iteration):
        _, _, g_loss, d_loss, train_top1 = sess.run([d_train_op, g_train_op,
                                                     'g_loss:0', 'd_loss:0', 'Accuracy/train_top1:0'])
        mean_d_loss += d_loss
        mean_g_loss += g_loss
        mean_top1 += train_top1

    print("mean_g_loss:", mean_g_loss / num_iteration)
    print("mean_d_loss:", mean_d_loss / num_iteration)
    print("mean_train_top1", mean_top1 / num_iteration)

    print("training Done!!!!!!!!!!!!!")
    print("-----------------------next")


def test_num_iteration(sess, test_logits, num_iteration=100):
    mean_top1 = 0.0
    print("testing ...............................global step ", GLOBAL_STEP)
    for i in range(num_iteration):
        test_top1, true_label, pred = sess.run(['Accuracy_1/test_top1:0', "test_batch_label:0", "test_pred:0"])
        mean_top1 += test_top1
        if test_top1 > 0.9:
            print("ground truth ", true_label)
            print("pred ", pred)
    mean_top1 /= float(num_iteration)

    print("mean_test_top1", mean_top1)
    print("test Done !!!!!!!!!!!!!")


def main():
    train_batch_sample, train_batch_label = readData.read_data(TRAIN_RECORDS, batch_size=64)
    train_batch_sample = tf.add(train_batch_sample,
                                tf.random_normal(shape=tf.shape(train_batch_sample), stddev=0.1))

    test_batch_sample, test_batch_label = readData.read_data(TEST_RECORDS, batch_size=64)

    train_batch_label_one_hot = tf.one_hot(train_batch_label, depth=config_.NUM_CLASSES, name="train_one_hot")
    test_batch_label_one_hot = tf.one_hot(test_batch_label, depth=config_.NUM_CLASSES)

    # build TSCGAN
    model = basic.TSCGAN()
    d_loss, g_loss = model.graph(train_batch_sample, train_batch_label_one_hot)

    train_top1, train_top5 = model.accuracy(model.logits_, train_batch_label_one_hot)

    d_train_op = model.get_opt(d_loss)
    g_train_op = model.get_opt(g_loss, D_or_G='G')

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
        writer = tf.summary.FileWriter(logdir='./ckpt', graph=sess.graph)
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                global GLOBAL_STEP
                # Run training steps or whatever
                train_num_iteration(sess=sess, d_train_op=d_train_op, g_train_op=g_train_op)
                test_num_iteration(sess=sess, test_logits=test_logits)
                d_saver.save(sess=sess, save_path="ckpt/%s.ckpt" % DataSetName, global_step=GLOBAL_STEP)
                GLOBAL_STEP += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print('interrupted...')
        finally:
            print("cleaning...")
            # When done, ask the threads to stop.
            coord.request_stop()

            # Wait for threads to finish.
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    main()
    pass
