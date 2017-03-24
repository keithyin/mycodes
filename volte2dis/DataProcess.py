#-*- coding:utf8 -*-
import pandas
import numpy as np
# 用 one hot的形式表示距离，即[1,0,0,0]表示0米，[0,1,0,0]表示0.5米,类推
def to_one_hot(vals,num_classes):
    res = np.zeros(shape=[len(vals),num_classes])
    for i,val in enumerate(vals):
        res[i,int(val)] = 1
    return np.array(res)
# MOdel的数据用以下两个函数处理，没有进行归一化
def data_process(file_name):
    file_name = "volte.csv"
    data = pandas.read_csv(file_name)
    data = np.array(data)
    data[data == '0m'] = 0
    data[data == '0.5m']=1
    data[data == '1m'] =2
    data[data=='1.5m'] = 3
    np.random.shuffle(data)
    return data
def get_data(file_name):
    data = pandas.read_csv(file_name)
    data = np.array(data)
    data = data.T
    data = data.reshape(-1,100)
    zeros = np.zeros(100)
    ones = np.ones(100)
    twos = np.ones(100)*2
    threes = np.ones(100)*3
    concat = np.concatenate([zeros,ones,twos,threes]).reshape(-1,1)
    concat = np.concatenate([data,concat],axis=1)
    return concat


def get_data2(file_name, size_per_example, stride):
    data = pandas.read_csv(file_name)
    data = np.array(data)
    data = data.T
    concat = np.array([0, 1, 2, 3]).reshape(-1, 1)
    ress = []
    for i in range(10000):
        if i * stride + size_per_example > 10000:
            break
        examples = data[:, i * stride:(i * stride + size_per_example)]
        res = np.concatenate([examples, concat], 1)
        ress.append(res)
    ress = np.concatenate(ress, 0)

    np.random.shuffle(ress)
    return ress

# ModelCNN_v1 和v2都是用的这个函数进行的数据处理
def get_data_with_normalization(file_name, size_per_example, stride):
    data = pandas.read_csv(file_name)
    data = np.array(data)
    data = data.T
    data = (data - np.min(data)) / (np.max(data)-np.min(data)) # 归一化处理
    data = data- np.mean(data) # 均值归零 处理
    concat = np.array([0, 1, 2, 3]).reshape(-1, 1)
    ress = []
    for i in range(data.shape[1]):
        if i * stride + size_per_example > data.shape[1]:
            break
        examples = data[:, i * stride:(i * stride + size_per_example)]
        res = np.concatenate([examples, concat], 1)
        ress.append(res)
    ress = np.concatenate(ress, 0)

    np.random.shuffle(ress)
    return ress
if __name__ == '__main__':
    file_name = "vol.csv"
    data = get_data_with_normalization(file_name,5,3)
    print(data[0:5])
