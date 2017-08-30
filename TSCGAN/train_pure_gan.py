import tensorflow as tf
from data import readData
from models import puregan
from config import cnf
import sys
import os
from tensorflow.python import debug as tf_debug

current_dir = sys.path[0]
config_ = cnf.Beef

GLOBAL_STEP = 0
TRAIN_RECORDS = config_.TRAIN_RECORDS
TEST_RECORDS = config_.TEST_RECORDS
DataSetName = config_.DATASET_NAME

tf.set_random_seed(2017)
NUM_TRAIN_ITERATION = config_.num_train_data // config_.BATCH_SIZE


def train_num_iteration(sess, d_train_op, g_train_op, num_iteration, model=None):
    mean_d_loss = 0.0
    mean_g_loss = 0.0
    for i in range(num_iteration):
        _, _, d_loss, g_loss = sess.run([d_train_op, g_train_op, 'd_loss:0', 'g_loss:0'],
                                        feed_dict={'bn_train:0': True})
        mean_d_loss += d_loss
        mean_g_loss += g_loss

    return mean_d_loss / num_iteration, mean_g_loss / num_iteration


def main():
    train_batch_sample, train_batch_label = readData.read_data(TRAIN_RECORDS, config=config_)

    # build TSCGAN

    model = puregan.GAN(config=config_)
    d_loss, g_loss = model.graph(train_batch_sample)

    d_train_op = model.get_opt(d_loss, regulizer=False)
    g_train_op = model.get_opt(g_loss, D_or_G='G', regulizer=False)

    gpu_conf = tf.GPUOptions(allow_growth=True)
    sess_conf = tf.ConfigProto(gpu_options=gpu_conf)
    with tf_debug.LocalCLIDebugWrapperSession(tf.Session(config=sess_conf)) as sess:
        writer = tf.summary.FileWriter(logdir=os.path.join(current_dir, 'ckpt'), graph=sess.graph)
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                global GLOBAL_STEP
                # Run training steps or whatever
                d_loss, g_loss = train_num_iteration(sess=sess, d_train_op=d_train_op, g_train_op=g_train_op,
                                                     num_iteration=NUM_TRAIN_ITERATION, model=model)
                model.decay_learning_rate()

                print("step %d, d loss %.6f , g loss %.6f" % (
                    GLOBAL_STEP * NUM_TRAIN_ITERATION, d_loss, g_loss))
                if GLOBAL_STEP % 100 == 0:
                    model.save_ckpt(save_path=os.path.join(current_dir,
                                                           "ckpt_pure_gan/GAN%s.ckpt" % DataSetName),
                                    global_step=GLOBAL_STEP)

                GLOBAL_STEP += 1
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        except KeyboardInterrupt:
            print('KeyboardInterrupt...................')
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
