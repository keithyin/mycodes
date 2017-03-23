#-*-coding:utf8-*-
import numpy as np
from scipy.misc import imread
import codecs
import os
import collections
import matplotlib.pyplot as plt
import gc
import time
_PAD=u'PAD'
_EOS=u'EOS'
_UNK=u'UNK'
_GO=u'GO'

PAD_ID=0
GO_ID=1
EOS_ID=2
UNK_ID=3
def all_txt_names(file_names):
    txt_names = []
    for names in file_names:
        txt_names.append(names[-1])
    return txt_names

def get_file_names(root_path):
    dirs = os.listdir(root_path)
    file_names = []
    for dir_name in dirs:
        second_path = root_path + "/" + dir_name
        second_dirs = os.listdir(second_path)
        for second_dir_name in second_dirs:
            third_path = second_path + "/" + second_dir_name
            third_dirs = os.listdir(third_path)
            for fourth in third_dirs:
                fourth_path = third_path + "/" + fourth
                file_names_per_sentenses = os.listdir(fourth_path)
                file_names_per_sentenses = sorted(file_names_per_sentenses)
                file_names_per_sentenses = [fourth_path + "/" + name for name in file_names_per_sentenses]
                file_names.append(file_names_per_sentenses)
    return file_names
def generate_dic(txt_names):
    counts = [(_PAD, -1), (_GO, -1), (_EOS, -1), (_UNK, -1)]
    words = []
    for name in txt_names:
        with codecs.open(name,encoding="gbk") as file:
            for line in file:
                words.extend(line[:-2])
                break
    counts.extend(collections.Counter(words).most_common())
    # print counts
    # exit()
    dic = {}
    for i,data in enumerate(counts):
        dic[data[0]] = i
    reversed_dic = dict(zip(dic.values(),dic.keys()))
    num_dic = len(dic)
    # print dic
    # exit()
    return dic, reversed_dic, num_dic

def file2id(file_name, dic):
    """
    :param file_name: string
    :param dic: dic {word:id}
    :return: ids
    """
    ids = []
    if not isinstance(file_name,str):
        raise ValueError("the type of file name must be string")
    with codecs.open(file_name,encoding="gbk") as file:
        for line in file:
            words = line[:-2]
            # print [words]
            # exit()
            for word in words:
                if word in dic.keys():
                    ids.append(dic[word])
                else:
                    ids.append(UNK_ID)
            break
    return ids
def img2arr(image_names):
    """
    images to arr  num*[height, width, channel]
    :param image_names: a list of image_names
    :return: a list of [height, width, channel]
    """
    data = []
    for name in image_names:
        image_data = imread(name)
        image_data = (image_data-np.mean(image_data))

        data.append(image_data)
    return data  #num * [height,width,channel]
def sentence_padding(data, needed_length):
    """
    for padding
    :param data: type list
    :param needed_length: int
    :return: padded data, no _GO symbol
    """
    ori_length = len(data)
    padding_size = needed_length - ori_length - 2
    pad_data=[]
    if padding_size > 0:
        pad_data = [PAD_ID]*padding_size
    data = [GO_ID]+data+[EOS_ID]+pad_data
    return data, ori_length
def pad_zero(x,needed_length):
    pad_length = needed_length - len(x)
    if pad_length > 0:
        x = list(x)+[0]*pad_length
    return x
class _BatchGenerator(object):
    def __init__(self, file_names, dic, batch_size=1):
        self.file_names = list(file_names) #num_data*[num_image_per_sentence]
        self.num_data = len(file_names)
        self.batch_size = batch_size
        self.dic = dic
        self.epoch_size = int(len(file_names)/self.batch_size)
        self.counter = 0
    def next(self):
        batch_x = [] # batch_size*[num_image_per_sen,height,weight,channel]
        batch_label=[] #batch_size * [21]
        data=self.file_names[self.counter*self.batch_size:(self.counter+1)*self.batch_size] #batch_size *[variable_length]
        weights = []
        for names in data:
            image_names = names[:-2]
            txt_name = names[-1]
            batch_x.append(img2arr(image_names))
            raw_ids = file2id(txt_name,self.dic)
            padded_ids,ori_length = sentence_padding(raw_ids,21)
            weight = pad_zero(np.ones(shape=ori_length),21)
            weights.append(weight)
            batch_label.append(padded_ids)
        weights = np.array(weights).T # shape: T *[batch_size]
        batch_label = np.array(batch_label)
        self.counter += 1
        return batch_x, batch_label, weights # (batch_size*[num_image_per_sen,height,weight,channel],b_s*[21])
    def status(self):
        if self.counter >= self.epoch_size:
            return False
        else:
            return True

class BatchGenerator(object):
    def __init__(self, file_names, dic, batch_size=1):
        self.bg = _BatchGenerator(file_names,dic, batch_size)
        self.buffer_size = 1000
        self.__buffer=[]
        self.buffer_counter = 0
        self.iter_counter = 0
        self.epoch_size = self.bg.epoch_size
        self.all_time = 0
        self.fill_buffer()
    def fill_buffer(self):
        print("filling buffer...")
        begin = time.time()
        counter = 0
        while counter < self.buffer_size and self.bg.status():
            counter += 1
            self.__buffer.append(self.bg.next())
        self.all_time += time.time() - begin
        print("consuming time : %.7f"%(time.time()-begin))
        print("processing samples %d-->%d"%(self.buffer_counter*self.buffer_size
                                            ,self.buffer_counter*self.buffer_size+
                                            len(self.__buffer)))
        self.buffer_counter += 1
        if len(self.__buffer) < self.buffer_size:
            print("all consuming time %.7f"%(self.all_time))
    def next(self):
        res = self.__buffer[self.iter_counter]
        self.iter_counter += 1
        if self.iter_counter >= self.buffer_size:
            del self.__buffer
            self.__buffer = []
            self.fill_buffer()
            self.iter_counter = 0
        return res
class BatchGenerator2(object):
    def __init__(self, file_names, dic, batch_size=1):
        self.bg = _BatchGenerator(file_names,dic, batch_size)
        self.buffer_size = 300
        self.buffer_counter = 0
        self.iter_counter = 0
        self.epoch_size = self.bg.epoch_size
        self.data_file_names = ["data_%d.npy"%i for i in range(4)]
        self.__data_counter = 0
    def load_data(self):
        self.__buffer = np.load(self.data_file_names[self.__data_counter])
        self.__data_counter += 1
        gc.collect()
    def next(self):
        if self.iter_counter >= len(self.__buffer):
            self.load_data()
        data = self.__buffer[self.iter_counter]
        self.iter_counter += 1
        return data


class ToDisk(object):
    def __init__(self, file_names, dic, batch_size=1):
        self.bg = _BatchGenerator(file_names,dic, batch_size)
        self.buffer_size = 300
        self.__buffer=[]
        self.buffer_counter = 0
        self.iter_counter = 0
        self.epoch_size = self.bg.epoch_size
        self.file_counter = 0
        while self.bg.status():
            begin=time.time()
            self.fill_buffer()
            print("saving to disk")
            print("time:%.7f"%(time.time()-begin))
            np.save("data_%d.npy"%self.file_counter,self.__buffer)
            print(len(self.__buffer))
            print("next round")
            self.buffer_counter += 1
            del self.__buffer
            gc.collect()
            self.__buffer = []

    def fill_buffer(self):
        counter = 0
        while counter < self.buffer_size and self.bg.status():
            counter += 1
            self.__buffer.append(self.bg.next())
            print(counter)


def split_set(all_file_names, rate = 0.1):
    length = len(all_file_names)
    train_set_length = int((1-rate) * length)
    test_set_length = length-train_set_length
    train_set = all_file_names[:-test_set_length]
    test_set = all_file_names[-test_set_length:]
    return train_set, test_set

if __name__ == "__main__":
    print("reading file names")
    root_path = "mouthregion_dataSet_from_yangfan"
    file_names = get_file_names(root_path)
    txt_names = all_txt_names(file_names)
    train_set, test_set = split_set(file_names)
    print("generating dic")
    dic, reversed_dic, count = generate_dic(txt_names)
    print("writing to disk")
    td = ToDisk(file_names,dic)



