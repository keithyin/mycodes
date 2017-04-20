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
restore_from_ckpt=False
global_step = 0
class Config(object):
    batch_size = 40
    num_units = 20
    learning_rate = 0.01
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
        self.true_label = tf.placeholder(dtype=tf.float32, shape=[None,4])
        self.predict = None
        self.loss = None
        self.loss_is_nan = None
        self.train_op = None
        self.accurate = None
        self.merged_summary = None
        self.graph()
    def graph(self):
        x = tf.reshape(self.data_in,shape=[-1,10,10])
        x = tf.expand_dims(x, -1)  # x:[None, 100, 1, 1]

        conv1W = tf.get_variable("conv1W",shape=[3,3,1,4],dtype=tf.float32)
        conv1b = tf.get_variable("conv1b",shape=[4],dtype=tf.float32)
        tf.summary.histogram("conv1W",conv1W)
        conv1 = tf.nn.conv2d(x,conv1W,[1,2,2,1],padding="SAME") # 第一层卷积

        conv1 = tf.nn.bias_add(conv1,conv1b)
        conv1 = tf.maximum(0.6*conv1,conv1)

        pool1 = tf.nn.avg_pool(conv1,[1,2,2,1],[1,2,2,1],padding="SAME") # pooling层
        tf.summary.histogram("pool1",pool1)
        conv2W = tf.get_variable("conv2W", shape=[3, 3, 4, 4], dtype=tf.float32)
        conv2b = tf.get_variable("conv2b", shape=[4], dtype=tf.float32)
        tf.summary.histogram("conv2W", conv2W)
        conv2 = tf.nn.conv2d(pool1, conv2W, [1, 2, 2, 1], padding="SAME") #第二层卷积
        conv2 = tf.nn.bias_add(conv2, conv2b)

        conv2 = tf.maximum(0.6*conv2,conv2)
        pool2 = tf.nn.avg_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")#pooling层
        tf.summary.histogram("pool2",pool2)
        # flatten = tf.reshape(pool2,shape=[-1,np.prod(pool2.get_shape()[1:])])
        flatten = tf.reshape(pool2, shape=[-1, np.prod(pool2.get_shape().as_list()[1:])]) #展平
        tf.summary.histogram("flatten",flatten)

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
        self.true_in = tf.arg_max(self.true_label,dimension=1)
        self.predict = tf.arg_max(softmax_output,dimension=1)
        self.accurate = tf.reduce_mean(tf.cast(tf.equal(self.predict,tf.arg_max(self.true_label,dimension=1)),tf.float32))
        # loss = tf.reduce_mean(-tf.reduce_sum((self.true_label*tf.log(softmax_output+ 1e-9)+
        #                                            (1-self.true_label)*tf.log(1-softmax_output + 1e-9)),1))
        # self.loss_is_nan = tf.is_nan(loss)
        # tf.summary.scalar("loss",loss)
        tf.summary.scalar("accurate",self.accurate)

        # opt = tf.train.GradientDescentOptimizer(self.config.learning_rate) #随机梯度下降优化
        #
        # vars = tf.trainable_variables()
        # self.grads = tf.gradients(loss, vars)
        # for grad,var in zip(self.grads, vars):
        #     tf.summary.histogram("grads_%s"%var.name, grad)
        # self.train_op = opt.apply_gradients(zip(self.grads,vars))
        #
        # self.merged_summary = tf.summary.merge_all()
        # self.loss = loss

def run_epoch(sess, model, data, batch_size,writer=None,saver=None):
    global global_step
    flag = False
    test_set_size = len(data)
    # train_data = data[:len(data) - test_set_size]
    # train_x = train_data[:, :-1]
    # train_label = DataProcess.to_one_hot(train_data[:, -1], 4)
    test_data = data[len(data) - test_set_size:]
    test_x = test_data[:, :-1]
    test_label = DataProcess.to_one_hot(test_data[:, -1], 4)

    # train

    # for i in range(int((len(data) - test_set_size) / batch_size)):
    #     global_step += 1
    #     feed_dic = {}
    #     feed_dic[model.data_in.name] = train_x[i * batch_size:(i + 1) * batch_size]
    #     feed_dic[model.true_label.name] = train_label[i * batch_size:(i + 1) * batch_size]
    #     _, accu, los, merged_summ, is_nan, fc1_, grads_= sess.run([model.train_op, model.accurate, model.loss,
    #                                                   model.merged_summary, model.loss_is_nan,
    #                                                        model.fc1,model.grads], feed_dic)
    #     if is_nan:
    #         print (train_x[i * batch_size:(i + 1) * batch_size])
    #         print (train_label[i * batch_size:(i + 1) * batch_size])
    #         exit()
    #     if global_step % 10000 == 0:
    #         flag = True
    #         # print "grads", grads_[0:2]
    #         # print "fc1:",fc1_[0:4,:]
    #         # print "output_",output_[0:4,:]
    #         # print "relu_output",relu_output_[0:4,:]
    #         print ("training:   "),
    #         print ("los:%.2f, accu:%.2f" % (los, accu))
    #         writer.add_summary(merged_summ, global_step // 10000)
    #         saver.save(sess, "checkpoint_cnn_v2/modelCNN")
    # test
    summ = np.zeros([4,4],dtype=np.int16)
    for j in range(int(test_set_size / batch_size)):
        feed_dic = {}
        feed_dic[model.data_in.name] = test_x[j * batch_size:(j + 1) * batch_size]
        feed_dic[model.true_label.name] = test_label[j * batch_size:(j + 1) * batch_size]
        actual, pred = sess.run([model.true_in, model.predict], feed_dic)
        for x,y in zip(actual,pred):
            summ[x,y] += 1

    print(summ)
if __name__ == '__main__':
    file_name = "vol1.csv"
    config = Config() # 用RNN文件中的Config, Config的batch_size 设置为测试样本的个数
    model = Model(config)# 将这个模型换成RNN model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, "checkpoint_cnn_v2/modelCNN")
        #writer = tf.summary.FileWriter("checkpoint_cnn_v2", sess.graph)
        print("processing data")
        data = DataProcess.get_data_with_normalization_1(file_name, 100, 20)
        print("number of data:%d"%len(data)) # batch_size就设成len(data)
        np.random.shuffle(data)
        run_epoch(sess, model, data, config.batch_size)
