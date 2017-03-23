#-*- coding:utf8 -*-
import preprocess
from preprocess import BatchGenerator
from config import Config
from config import LSTMConfig
from pylab import *
import tensorflow as tf
import time
import numpy as np
import Levenshtein
import os
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
xrange = range
global_step = 0
def conv2d(inputs, kernel, biases, s_h, s_w):
    convolve = lambda i, k: tf.nn.conv2d(i,k,strides=[1,s_h,s_w,1],padding="SAME")
    conv = convolve(inputs, kernel)
    return tf.nn.bias_add(conv, biases)
get_conv_bias = lambda c_o, name:tf.get_variable(name,shape=[c_o],dtype=tf.float32)
get_fc_w = lambda i, o, name:tf.get_variable(name, shape=[i,o],dtype=tf.float32)
get_fc_bias = lambda o, name: tf.get_variable(name, shape=[o], dtype=tf.float32)
def get_conv_w(k_h, k_w, in_channels, out_channels,name):
    conv_w = tf.get_variable(name=name, shape=[k_h,k_w,in_channels,out_channels])
    return conv_w
def arr2str(arr):
    string = "".join(c for c in arr)
    return string
class Model(object):
    def __init__(self, scope, is_training=True):
        self.name = 'i_am_model'
        self.initial_state = None
        self.final_state = None
        self.predict = None
        self.encoder_inputs = tf.placeholder(tf.float32,shape=[None,227,227,3], name="encoder_inputs")
        self.decoder_inputs = []
        self.target_weights = []
        self.accuracy = 0
        # for i in xrange(LSTMConfig.lstm_max_time):
        #     self.encoder_inputs.append(tf.placeholder(tf.float32,shape=[None, 1024],name="encoder_inputs%d"%i))
        for i in xrange(LSTMConfig.lstm_max_time+1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="decoder%d"%i))
            self.target_weights.append(tf.placeholder(tf.float32,shape=[None],name="target_weights%d"%i))
        self.targets = [self.decoder_inputs[i+1] for i in xrange(len(self.decoder_inputs)-1)]
        outputs = self.alex_net2(self.encoder_inputs, is_training)  # [None, 4096]

        assert len(outputs.get_shape()) == 2
        outputs = tf.reduce_mean(outputs, 0)
        outputs = tf.squeeze(outputs)
        lstm_in = tf.expand_dims(outputs, 0)
        lstm_in = [lstm_in] * 20
        # convert list to tensor
        lstm_in = tf.convert_to_tensor(lstm_in)

        loss = self.lstm_units(lstm_in, is_training, self.decoder_inputs)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", loss)
        vars = tf.trainable_variables()
        grads = tf.gradients(loss,vars)
        for grad, var in zip(grads,vars):
            tf.summary.histogram(var.name, var)
            tf.summary.histogram("grad_%s"%var.name,grad)

        self.merged_summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope=scope))
        self.loss_is_nan = tf.is_nan(loss)
        self.train_op = tf.train.GradientDescentOptimizer(Config.initial_learning_rate).minimize(loss)
        self.loss = loss
        tf.add_to_collection("loss", self.loss)
    def alex_net2(self,inputs, is_traning,targets=None):
        """
        alex net
        :param inputs: [None, 227, 227, 3]
        :param is_traning:
        :param targets:
        :return: [None, 4096]
        """
        def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
            '''
            From https://github.com/ethereon/caffe-tensorflow
            '''
            c_i = input.get_shape()[-1]
            assert c_i % group == 0
            assert c_o % group == 0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(input, group, 3)
                kernel_groups = tf.split(kernel, group, 3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat_v2( output_groups,3)
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11;k_w = 11;c_o = 96;s_h = 4;s_w = 4;
        conv1W = tf.get_variable("conv1W", shape=[11,11,3,96],dtype=tf.float32)
        conv1b = tf.get_variable("conv1b", shape=[96],dtype=tf.float32)
        conv1_in = conv(inputs, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2;alpha = 2e-05;beta = 0.75;bias = 1.0;
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3;k_w = 3;s_h = 2;s_w = 2;
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5;k_w = 5;c_o = 256;s_h = 1;s_w = 1;group = 2;
        conv2W = tf.get_variable("conv2W", shape=[5,5,48,256])
        conv2b = tf.get_variable("conv2b", shape=[256],dtype=tf.float32)
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2;alpha = 2e-05;beta = 0.75;bias = 1.0;
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3;k_w = 3;s_h = 2;s_w = 2;
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3;k_w = 3;c_o = 384;s_h = 1;s_w = 1;group = 1;
        conv3W = tf.get_variable("conv3W",shape=[3,3,256,384])
        conv3b = tf.get_variable("conv3b",shape=[384])
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3;c_o = 384;s_h = 1;s_w = 1;group = 2;
        conv4W = tf.get_variable("conv4W", shape=[3,3,192,384])
        conv4b = tf.get_variable("conv4b", shape=[384])
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3;k_w = 3;c_o = 256;s_h = 1;s_w = 1;group = 2;
        conv5W = tf.get_variable("conv5W", shape=[3,3,192,256])
        conv5b = tf.get_variable("conv5b", shape=[256])
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3;k_w = 3;s_h = 2;s_w = 2;
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # fc6
        # fc(4096, name='fc6')
        fc6W = tf.get_variable("fc6W", shape=[int(prod(maxpool5.get_shape()[1:])),4096])
        fc6b = tf.get_variable("fc6b", shape=[4096])
        fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)


        fc7W = tf.get_variable("fc7W", shape=[4096,4096])
        fc7b = tf.get_variable("fc7b",shape=[4096])
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)


        conv1_image = conv1W[:, :, 0, 0]
        conv1_image = tf.expand_dims(conv1_image, dim=0)
        conv1_image = tf.expand_dims(conv1_image, dim=-1)
        feature_map_1 = conv1[:, :, :, 0]
        feature_map_1 = tf.expand_dims(feature_map_1, dim=-1)
        tf.summary.image("image", conv1_image)
        tf.summary.image("image1layer", feature_map_1)

        return fc7
    def lstm_units(self, inputs, is_training, decoder_inputs):
        """
        :param inputs: Tensor[T,batch_size, size]
        :param decoder_inputs:T*[batch_size]
        :return:
        """

        lstm_cell = rnn.LSTMCell(LSTMConfig.num_unit,state_is_tuple=True)
        if is_training:
            lstm_cell = rnn.DropoutWrapper(lstm_cell,output_keep_prob=0.5)
        lstm_cell = rnn.MultiRNNCell(cells=[lstm_cell], state_is_tuple=True)
        initial_state = lstm_cell.zero_state(1,dtype=tf.float32)
        output_projection_w = tf.get_variable("projection_w",shape=[Config.output_embedding_size,
                                                                    Config.vocab_size],dtype=tf.float32)
        output_projection_b = tf.get_variable("projection_b", shape=[Config.vocab_size],dtype=tf.float32)
        output_projection = (output_projection_w,output_projection_b)
        (outputs, en_state) = tf.nn.dynamic_rnn(lstm_cell,inputs,initial_state=initial_state,dtype=tf.float32,
                                                time_major=True, scope="encode")
        outputs = tf.reshape(tf.concat_v2(outputs,1),shape=[-1,20,LSTMConfig.num_unit])
        if is_training:
            (de_outputs,de_state)=legacy_seq2seq.embedding_attention_decoder(decoder_inputs[:20],en_state,outputs,lstm_cell,
                                                  Config.vocab_size,
                                                  Config.output_embedding_size,
                                                  output_projection=output_projection
                                                  )
        else:
            (de_outputs, de_state) = legacy_seq2seq.embedding_attention_decoder(decoder_inputs[:20], en_state, outputs,
                                                                               lstm_cell,
                                                                               Config.vocab_size,
                                                                               Config.output_embedding_size,
                                                                               output_projection=output_projection,
                                                                               feed_previous=True
                                                                               )

        logits = []
        for output in de_outputs:
            logits.append(tf.matmul(output,output_projection[0])+output_projection[1])
        self.predict = tf.cast(tf.argmax(tf.squeeze(logits),1),tf.int64)
        tf.add_to_collection("predict", self.predict)
        loss = legacy_seq2seq.sequence_loss_by_example(logits,self.targets[:20],self.target_weights[:20],
                                              average_across_timesteps=True)

        return loss
def run_epoch(session, m, reversed_dic,summary_writer,dic,file_names,is_train_set=True):
    global global_step
    gen_batch = BatchGenerator(file_names, dic)
    epoch_accuracy = []
    losses = []
    begin_time = time.time()
    counter = 0
    co = gen_batch.epoch_size / 100
    for i in xrange(gen_batch.epoch_size): #gen_batch.train_set_size
        if i % co == 0:
            counter += 1
            print("-%d-"%counter),
        batch_x, decoder_input,target_weight = gen_batch.next()
        input_feed = {}
        input_feed[m.encoder_inputs.name] = batch_x[0]

        for i in xrange(Config.lstm_max_time+1):
            input_feed[m.decoder_inputs[i].name] = decoder_input[:, i]
            input_feed[m.target_weights[i].name] = target_weight[i]
        if is_train_set:
            loss,_,predict_ ,is_nan, summaries= session.run([m.loss, m.train_op,
                                                         m.predict, m.loss_is_nan,m.merged_summary],input_feed)
        else:
            loss, predict_, is_nan, summaries = session.run([m.loss, m.predict,
                                                                 m.loss_is_nan, m.merged_summary],input_feed)
        target = decoder_input[0]
        for j, label in enumerate(target):
            if label == 2 or j == 20:
                break
        pre_sen = [reversed_dic[index] for index in predict_[:j]]
        true_sen = [reversed_dic[index] for index in target[:j]]
        pre_str = arr2str(pre_sen)
        true_str = arr2str(true_sen)
        dis = Levenshtein.distance(pre_str,true_str)
        accuracy = float(dis)/len(true_str) # error rate
        epoch_accuracy.append(accuracy)
        losses.append(loss)

        if is_train_set:
            global_step += 1
        if is_train_set and global_step % 2000 == 0:
            m.saver.save(session,"checkpoint/model",global_step=i)
            summary_writer.add_summary(summaries, global_step)
    print("")
    end_time = time.time()
    print("time:%.7f" % (end_time - begin_time))
    if is_train_set:
        print("train, avg_loss: %.5f, avg_accuracy: %.5f" % (np.mean(losses),1-np.mean(epoch_accuracy)))
    else:
        print("test, avg_loss: %.5f, avg_accuracy: %.5f" % (np.mean(losses),1-np.mean(epoch_accuracy)))
    for id in predict_:
        print(reversed_dic[id]),
    print("")

def main(_):
    root_path = "mouthregion_dataSet_from_yangfan"
    file_names = preprocess.get_file_names(root_path)
    txt_names = preprocess.all_txt_names(file_names)
    train_set, test_set = preprocess.split_set(file_names)
    dic, reversed_dic,count = preprocess.generate_dic(txt_names)
    with tf.Graph().as_default():
        with tf.name_scope("train") as train_scope:
            model = Model(train_scope)
        with tf.name_scope("test") as test_scope:
            tf.get_variable_scope().reuse_variables()
            test_model = Model(scope=test_scope,is_training=False)
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            saver.restore(sess, "checkpoint/model.ckpt-99")
            summary_writer = tf.summary.FileWriter("checkpoint", sess.graph)
            for epoch in xrange(1):
                print("epoch:%d" % epoch)
                #run_epoch(sess, model, reversed_dic,summary_writer,dic,train_set)
                run_epoch(sess,test_model,reversed_dic,summary_writer,dic,test_set,is_train_set=False)
if __name__ == "__main__":
    tf.app.run()
