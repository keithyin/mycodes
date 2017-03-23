#-*- coding:utf8 -*-
import ModelV4 as Model
import tensorflow as tf
import preprocess
from preprocess import BatchGenerator
from config import Config
from config import LSTMConfig
from pylab import *
import time
import numpy as np
import Levenshtein
import os
import time
xrange = range
NUM_GPUS = 4
def avrage_gradients(gradients):
    """
    :param gradients: [[grad0, grad1,..], [grad0, grad1,..], [grad0, grad1, ...]]
    :return: avaged_gradients [grad0 ,grad1, grad2]
    """
    averaged_grads = []
    for grads_per_var in zip(*gradients):# grads_per_var: [grad_i, grad_i, ...]
        grads = []
        for grad in grads_per_var:
            expanded_grad = tf.expand_dims(grad, 0)
            grads.append(expanded_grad)
        grads = tf.concat(grads, 0)
        grads = tf.reduce_mean(grads, 0)
        averaged_grads.append(grads)
    return averaged_grads


def multi_gpu_model(num_gpus=1):
    # with tf.device("/cpu:0"):
    #     opt = tf.train.GradientDescentOptimizer(0.01)
    grads = []
    for i in xrange(num_gpus):
        with tf.device("/gpu:%d"%i):
            with tf.name_scope("tower_%d"%i) as scope:
                m = Model.Model(scope)
                tf.add_to_collection("train_model", m) # adding to collection can feed them easily
                tf.get_variable_scope().reuse_variables()
                grad = tf.gradients(m.loss, tf.trainable_variables())
                grads.append(grad)
    averaged_grads = avrage_gradients(grads)
    with tf.device("/cpu:0"):
        opt = tf.train.GradientDescentOptimizer(0.01)
        train_op = opt.apply_gradients(zip(averaged_grads, tf.trainable_variables()))
    return train_op
def generate_feed_dic(model,input_feed, batch_generator):
    batch_x, decoder_input, target_weight = batch_generator.next()

    input_feed[model.encoder_inputs.name] = batch_x[0]

    for i in xrange(Config.lstm_max_time + 1):
        input_feed[model.decoder_inputs[i].name] = decoder_input[:, i]
        input_feed[model.target_weights[i].name] = target_weight[i]


def run_epoch(session,data_set,dic, train_op=None, is_training=True):
    gen_batch = BatchGenerator(data_set, dic)
    losses = []
    counter = 0
    begin = time.time()
    for step in xrange(gen_batch.epoch_size // NUM_GPUS):
        if is_training:
            models = tf.get_collection("train_model")
            input_feed = {}
            for model in models:
                generate_feed_dic(model,input_feed,gen_batch)
            losss = tf.get_collection("loss")

            fetch_list = [item for item in losss]
            fetch_list.append(train_op)
            *lossss, _ = session.run(fetch_list, input_feed)
            #*predits, = session.run(tf.get_collection("predict"), input_feed)
            losses.extend(lossss)
    print("")
    print("loss:%.7f"%np.mean(losses))
    print("time: %.1f" % (time.time() - begin))
    return np.mean(losses)

def main(_):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    root_path = "mouthregion_dataSet_from_yangfan"
    print("preparing file names...")
    file_names = preprocess.get_file_names(root_path)
    txt_names = preprocess.all_txt_names(file_names)
    train_set, test_set = preprocess.split_set(file_names)
    train_set = sorted(train_set, key=len)
    dic, reversed_dic, count = preprocess.generate_dic(txt_names)
    print("done!")
    print("building model...")
    with tf.Graph().as_default():
        with tf.name_scope("train") as train_scope:
            train_op = multi_gpu_model(NUM_GPUS)

        #with tf.name_scope("test") as test_scope:
            #tf.get_variable_scope().reuse_variables()
            #test_model = Model.Model(scope=test_scope, is_training=False)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        print("allocating resources...")
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init)
            losses = []
            print("restoring variables...")
            saver.restore(sess,"checkpoint_v4/model_v4.ckpt")
            print("done")
            # if os.path.exists("checkpoint/model.ckpt"):
            #     model.saver.restore(sess, "checkpoint/model.ckpt")
            #summary_writer = tf.summary.FileWriter("checkpoint_v4", sess.graph)
            print("running epoch...")
            for epoch in xrange(1000):
                print("epoch:%d" % epoch)
                losses.append(run_epoch(sess,train_set,dic, train_op))
                saver.save(sess, "checkpoint_v4/model_v4.ckpt")
                if os.path.exists("losses.npy"):
                    os.remove("losses.npy")
                np.save("losses.npy",losses)
if __name__ == "__main__":
    tf.app.run()


