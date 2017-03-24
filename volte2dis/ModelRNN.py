#-*- coding:utf8 -*-
"""
数据处理：
100 个为一组， stride为20
RNN：
使用LSTM作为RNN单元，用于压缩时序信息。
结构：2层LSTM，时序展开为100， LSTM的单元为50，将最后一个时间的output输出给全连接层
全连接层：
第一层：20个神经元
输出层：4个神经元，然后接softmax输出
损失函数：cross entropy ， 优化算法 ：随机梯度下降
"""
import tensorflow as tf
import DataProcess
import numpy as np
import os
from tensorflow.contrib import rnn
global_step = 0
xrange = range
def leaky_relu(x, alpha = 0.6, min=-6, max=6):
    x = tf.maximum(alpha*x, x)
    if min is not None and max is not None:
        x = tf.clip_by_value(x, min, max)
    return x
class Config(object):
    batch_size = 10
    num_units = 20
    learning_rate = 0.002
class Model(object):
    def __init__(self, config):
        self.data_in = tf.placeholder(dtype=tf.float32,shape=[None,100])
        self.true_label = tf.placeholder(dtype=tf.float32, shape=[None,4])
        self.config = config
        self.predict = None
        self.accuracy = None
        self.loss = None
        self.loss_is_nan=None
    def graph(self):
        x = tf.expand_dims(self.data_in,-1) #[None, 100, 1]
        cell = rnn.LSTMCell(num_units=50,activation=leaky_relu)  # 使用LSTM单元
        cell = rnn.MultiRNNCell(cells=[cell]) #使用两层LSTM
        zero_states = cell.zero_state(self.config.batch_size,dtype=tf.float32)
        (output, state) = tf.nn.dynamic_rnn(cell,x,initial_state=zero_states,dtype=tf.float32)

        output = output[:,-1,:] #获取最后一个时刻的输出

        fc1W = tf.get_variable("fc1W",shape=[50,20],dtype=tf.float32)
        fc1b = tf.get_variable("fc1b",shape=[20],dtype=tf.float32)
        fc1 = leaky_relu(tf.add(tf.matmul(output,fc1W),fc1b)) #第一层全连接，激活函数使用ReLU

        fc2W = tf.get_variable("fc2W",shape=[20,4],dtype=tf.float32)
        fc2b = tf.get_variable("fc2b",shape=[4],dtype=tf.float32)
        fc2 = leaky_relu(tf.add(tf.matmul(fc1,fc2W),fc2b)) #第二层全连接，激活函数使用ReLU
        softmax_output = tf.nn.softmax(fc2) #接softmax输出

        self.predict = tf.arg_max(softmax_output, dimension=1)
        self.accurate = tf.reduce_mean(
            tf.cast(tf.equal(self.predict, tf.arg_max(self.true_label, dimension=1)), tf.float32))
        self.loss = tf.reduce_mean(-tf.reduce_sum((self.true_label * tf.log(softmax_output) +
                                                   (1 - self.true_label) * tf.log(1 - softmax_output)), 1))
        self.loss_is_nan = tf.is_nan(self.loss)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accurate", self.accurate)

        opt = tf.train.GradientDescentOptimizer(self.config.learning_rate)  # 随机梯度下降优化
        vars = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, vars)
        for grad, var in zip(self.grads, vars):
            tf.summary.histogram("grads_%s" % var.name, grad)
        self.train_op = opt.apply_gradients(zip(self.grads, vars))

        self.merged_summary = tf.summary.merge_all()
def run_epoch(sess, model, data, batch_size,writer,saver):
    global global_step
    flag = False

    test_set_size = int(len(data) // 10)
    train_data = data[:len(data)-test_set_size]
    test_data = data[len(data)-test_set_size:]
    train_x = train_data[:,:-1]
    train_label = DataProcess.to_one_hot(train_data[:,-1],4)
    test_x = test_data[:,:-1]
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
            flag=True
            print("training:"),
            print("los:%.7f, accu:%.7f" % (los,accu))
            writer.add_summary(merged_summ, global_step // 1000)
            saver.save(sess,"checkpoint_rnn/modelRNN.ckpt")
    #test
    test_losses=[]
    test_accuracy=[]
    for j in range(test_set_size//batch_size):
        feed_dic = {}
        feed_dic[model.data_in.name] = test_x[j*batch_size:(j+1)*batch_size]
        feed_dic[model.true_label.name] = test_label[j*batch_size:(j+1)*batch_size]
        accu, los= sess.run([model.accurate, model.loss], feed_dic)
        test_accuracy.append(accu)
        test_losses.append(los)
    if flag:
        print("test:"),
        print("los:%.2f, accu:%.2f" % (np.mean(test_losses), np.mean(test_accuracy)))
        flag = False

if __name__ == '__main__':
    file_name = "vol.csv"
    config = Config()
    model = Model(config)
    model.graph()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        if not os.path.exists("checkpoint_rnn"):
            os.mkdir("checkpoint_rnn")
        if os.path.exists("checkpoint_rnn/modelRNN.ckpt"):
            saver.restore(sess,"checkpoint_rnn/modelRNN.ckpt")
        writer = tf.summary.FileWriter("checkpoint_rnn",sess.graph)
        data = DataProcess.get_data_with_normalization(file_name,100,20)
        for epoch in xrange(100000):
            np.random.shuffle(data)
            print("epoch:%d" % epoch)
            run_epoch(sess,model,data,config.batch_size,writer,saver)

