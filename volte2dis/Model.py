#-*- coding:utf8 -*-
"""
数据处理：
将每行16个数据reshape成4*4的矩阵，将处理过的数据输入到卷积神经网络中。
卷积神经网络：
第一层：kernel:shape[2*2], stride[1,1]， pooling:shape[2*2],stride[2,2], avg pool
第二层:kernel:shape[2*2], stride[1,1]， pooling:shape[2*2],stride[2,2], avg pool
将pooling后的输出展平会得到[batch_size, 4]的矩阵
全连接层：
第一层： 10个神经元
输出层：4个神经元，然后接softmax输出
损失函数：cross entropy ， 优化算法 ：随机梯度下降
"""
import tensorflow as tf
import numpy as np
import DataProcess
import os
global_step = 0
xrange = range
class Model(object):
    def __init__(self):
        self.data_in = tf.placeholder(dtype=tf.float32,shape=[None, 4, 4])
        self.true_label = tf.placeholder(dtype=tf.float32, shape=[None,4])
        self.predict = None
        self.loss = None
        self.loss_is_nan = None
        self.train_op = None
        self.accurate = None
        self.merged_summary = None
    def graph(self):
        x = tf.expand_dims(self.data_in,-1)
        conv1W = tf.get_variable("conv1W",shape=[2,2,1,4],dtype=tf.float32)
        conv1b = tf.get_variable("conv1b",shape=[4],dtype=tf.float32)
        conv1 = tf.nn.conv2d(x,conv1W,[1,1,1,1],padding="SAME") #第一层卷积
        conv1 = tf.nn.bias_add(conv1,conv1b)
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.avg_pool(conv1,[1,2,2,1],[1,2,2,1],padding="SAME") # pooling层

        conv2W = tf.get_variable("conv2W", shape=[2, 2, 4, 4], dtype=tf.float32)
        conv2b = tf.get_variable("conv2b", shape=[4], dtype=tf.float32)
        conv2 = tf.nn.conv2d(pool1, conv2W, [1, 1, 1, 1], padding="SAME") #第二层卷积
        conv2 = tf.nn.bias_add(conv2, conv2b)
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.avg_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")#pooling层

        # flatten = tf.reshape(pool2,shape=[-1,np.prod(pool2.get_shape()[1:])])
        flatten = tf.reshape(pool2, shape=[-1, 4]) #展平

        fc1W = tf.get_variable("fc1W",shape=[np.prod(pool2.get_shape()[1:]),10],dtype=tf.float32)
        fc1b = tf.get_variable("fc1b",shape=[10],dtype=tf.float32)
        fc1 = tf.add(tf.matmul(flatten,fc1W),fc1b) # 第一层全连接
        fc1 = tf.nn.relu(fc1)
        outputW = tf.get_variable("outputW",shape=[10,4],dtype=tf.float32)
        outputb = tf.get_variable("outputb",shape=[4],dtype=tf.float32)
        output = tf.add(tf.matmul(fc1,outputW),outputb)
        output = tf.clip_by_value(tf.maximum(0.6*output, output), -6, 6)#输出层
        softmax_output = tf.nn.softmax(output)#输出接softmax

        self.predict = tf.arg_max(softmax_output,dimension=1)
        self.accurate = tf.reduce_mean(tf.cast(tf.equal(self.predict,tf.arg_max(self.true_label,dimension=1)),tf.float32))
        self.loss = tf.reduce_mean(-tf.reduce_sum((self.true_label*tf.log(softmax_output)+
                                                   (1-self.true_label)*tf.log(1-softmax_output)),1))
        self.loss_is_nan = tf.is_nan(self.loss)
        tf.summary.scalar("loss",self.loss)
        tf.summary.scalar("accurate",self.accurate)
        self.train_op = tf.train.GradientDescentOptimizer(0.005).minimize(self.loss) #随机梯度下降优化
        self.merged_summary = tf.summary.merge_all()
def run_epoch(sess, model, file_name, batch_size,writer,saver):
    global global_step
    flag = False
    data = DataProcess.data_process(file_name)
    test_set_size = int(len(data) // 10)
    train_data = data[:len(data)-test_set_size]
    test_data = data[len(data)-test_set_size:]
    train_x = train_data[:,:-1].reshape(-1,4,4)
    train_label = DataProcess.to_one_hot(train_data[:,-1],4)
    test_x = test_data[:,:-1].reshape(-1,4,4)
    test_label = DataProcess.to_one_hot(test_data[:,-1],4)

    #train

    for i in xrange((len(data)-test_set_size)//batch_size):
        global_step += 1
        feed_dic = {}
        feed_dic[model.data_in.name] = train_x[i*batch_size:(i+1)*batch_size]
        feed_dic[model.true_label.name] = train_label[i*batch_size:(i+1)*batch_size]
        _,accu,los,merged_summ, is_nan= sess.run([model.train_op,model.accurate,model.loss,
                                           model.merged_summary,model.loss_is_nan],feed_dic)
        if is_nan:
            print(train_x[i*batch_size:(i+1)*batch_size])
            print(train_label[i * batch_size:(i + 1) * batch_size])
            exit()
        if global_step % 1000 == 0:
            flag = True
            print("training:")
            print("los:%.7f, accu:%.7f" % (los,accu))
            writer.add_summary(merged_summ, global_step)
            saver.save(sess,"checkpoint/model")
    #test

    feed_dic = {}
    feed_dic[model.data_in.name] = test_x
    feed_dic[model.true_label.name] = test_label
    accu, los= sess.run([model.accurate, model.loss], feed_dic)
    if flag:
        flag = False
        print("test:")
        print("los:%.7f, accu:%.7f" % (los, accu))

if __name__ == '__main__':
    file_name = "volte.csv"
    model = Model()
    model.graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        if os.path.exists("checkpoint/model"):
            saver.restore(sess,"checkpoint/model")
        writer = tf.summary.FileWriter("checkpoint",sess.graph)
        for epoch in xrange(100000):
            print("epoch:%d" % epoch)
            run_epoch(sess,model,file_name,10,writer,saver)
