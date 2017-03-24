#-*- coding:utf8 -*-
"""
数据处理：
100 个为一组， stride为20 100转成10*10
卷积神经网络：
第一层：kernel:shape[4*1], stride[1,1]， pooling:shape[4*1],stride[4,1], avg pool
第二层:kernel:shape[4*1], stride[1,1]， pooling:shape[4*1],stride[4,1], avg pool
将pooling后的输出展平会得到[batch_size, M]的矩阵
全连接层：
第一层： M个神经元
输出层：4个神经元，然后接softmax输出
损失函数：cross entropy ， 优化算法 ：随机梯度下降

problem:
gradient vanishing
"""
import tensorflow as tf
import DataProcess
import numpy as np
import os
global_step = 0
MAP_DIC=[0, 0.5, 1, 1.5]
class Config(object):
    batch_size = 40
    num_units = 20
    learning_rate = 0.1
def kernel2img(tensor):
    res = []
    for i in range(tensor.get_shape().as_list()[-2]):
        for j in range(tensor.get_shape().as_list()[-1]):
            res.append(tensor[:,:,i,j])
    res = tf.convert_to_tensor(res)
    res = tf.expand_dims(res,-1)
    return res
class Model(object):
    def __init__(self,config):
        self.config = config
        self.data_in = tf.placeholder(dtype=tf.float32,shape=[None, 100])
        #self.true_label = tf.placeholder(dtype=tf.float32, shape=[None,4])
        self.predict = None
        self.loss = None
        self.loss_is_nan = None
        self.train_op = None
        self.accurate = None
        #self.merged_summary = None
        self.graph()
    def graph(self):
        x = tf.reshape(self.data_in,shape=[-1,10,10])
        x = tf.expand_dims(x, -1)  # x:[None, 100, 1, 1]

        conv1W = tf.get_variable("conv1W",shape=[3,3,1,4],dtype=tf.float32)
        conv1b = tf.get_variable("conv1b",shape=[4],dtype=tf.float32)
        #tf.summary.histogram("conv1W",conv1W)
        conv1 = tf.nn.conv2d(x,conv1W,[1,2,2,1],padding="SAME") # 第一层卷积

        conv1 = tf.nn.bias_add(conv1,conv1b)
        conv1 = tf.maximum(0.6*conv1,conv1)

        pool1 = tf.nn.avg_pool(conv1,[1,2,2,1],[1,2,2,1],padding="SAME") # pooling层
        #tf.summary.histogram("pool1",pool1)
        conv2W = tf.get_variable("conv2W", shape=[3, 3, 4, 4], dtype=tf.float32)
        conv2b = tf.get_variable("conv2b", shape=[4], dtype=tf.float32)
        #tf.summary.histogram("conv2W", conv2W)
        conv2 = tf.nn.conv2d(pool1, conv2W, [1, 2, 2, 1], padding="SAME") #第二层卷积
        conv2 = tf.nn.bias_add(conv2, conv2b)

        conv2 = tf.maximum(0.6*conv2,conv2)
        pool2 = tf.nn.avg_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")#pooling层
        #tf.summary.histogram("pool2",pool2)
        # flatten = tf.reshape(pool2,shape=[-1,np.prod(pool2.get_shape()[1:])])
        flatten = tf.reshape(pool2, shape=[-1, np.prod(pool2.get_shape().as_list()[1:])]) #展平
        #tf.summary.histogram("flatten",flatten)

        fc1W = tf.get_variable("fc1W",shape=[np.prod(pool2.get_shape()[1:]),4],dtype=tf.float32)
        fc1b = tf.get_variable("fc1b",shape=[4],dtype=tf.float32)
        fc1 = tf.add(tf.matmul(flatten,fc1W),fc1b) # 第一层全连接
        fc1 = tf.clip_by_value(tf.maximum(0.6*fc1,fc1),-6,6)
        self.fc1 = fc1
        # outputW = tf.get_variable("outputW",shape=[6,4],dtype=tf.float32)
        # outputb = tf.get_variable("outputb",shape=[4],dtype=tf.float32)
        # output = tf.add(tf.matmul(fc1,outputW),outputb)
        # self.output = output
        # relu_output = tf.nn.relu6(output) #输出层
        softmax_output = tf.nn.softmax(fc1)#输出接softmax
        # self.relu_output = relu_output
        self.predict = tf.arg_max(softmax_output,dimension=1)
        # self.accurate = tf.reduce_mean(tf.cast(tf.equal(self.predict,tf.arg_max(self.true_label,dimension=1)),tf.float32))
        # loss = tf.reduce_mean(-tf.reduce_sum((self.true_label*tf.log(softmax_output+ 1e-9)+
        #                                            (1-self.true_label)*tf.log(1-softmax_output + 1e-9)),1))
        # self.loss_is_nan = tf.is_nan(loss)
        # tf.summary.scalar("loss",loss)
        # tf.summary.scalar("accurate",self.accurate)
        # opt = tf.train.GradientDescentOptimizer(self.config.learning_rate) #随机梯度下降优化
        #
        # vars = tf.trainable_variables()
        # self.grads = tf.gradients(loss, vars)
        # for grad,var in zip(self.grads, vars):
        #     tf.summary.histogram("grads_%s"%var.name, grad)
        # self.train_op = opt.apply_gradients(zip(self.grads,vars))

        # self.merged_summary = tf.summary.merge_all()
        # self.loss = loss

def predict(sess, model, data):
    data_in = data
    predict_ = sess.run(model.predict, feed_dict={model.data_in.name:data_in})
    return predict_
if __name__ == '__main__':
    file_name = "vol2.csv"
    config = Config()
    model = Model(config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "checkpoint_cnn_v2/modelCNN")
        data = DataProcess.process_data(file_name)
        res = predict(sess,model,data)
        for i,data in enumerate(res):
            print("%d, dis:%.1f"%(i,MAP_DIC[data]))