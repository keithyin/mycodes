#-*- coding:utf8 -*-
import numpy as np
import pandas
class BatchGenerator(object):
    def __init__(self,file_name,batch_size,size_per_example,stride):
        self.file_name = file_name
        self.batch_size = batch_size
        self.size_per_example = size_per_example
        self.stride = stride
        self.ress = None
        self.counter = 0
        self.train_set = None
        self.test_set = None
        self.num_batch = None
    def get_data(self):
        data = pandas.read_csv(self.file_name)
        data = np.array(data)
        data = data.T
        concat = np.array([0,1,2,3]).reshape(-1,1)
        ress = []
        for i in range(10000):
            if i*self.stride+self.size_per_example > 10000:
                break
            examples=data[:,i*self.stride:(i*self.stride+self.size_per_example)]
            res = np.concatenate([examples,concat],1)
            ress.append(res)
        ress = np.concatenate(ress, 0)
        self.ress = ress
        np.random.shuffle(self.ress)
        self.train_set = self.ress[:, :-self.batch_size]
        self.num_batch = len(self.train_set) / self.batch_size
        self.test_set = self.ress[:, -self.batch_size:]
        # print ress.shape
        # data = data.reshape(-1, 100)
        # zeros = np.zeros(100)
        # ones = np.ones(100)
        # twos = np.ones(100) * 2
        # threes = np.ones(100) * 3
        # concat = np.concatenate([zeros, ones, twos, threes]).reshape(-1, 1)
        # concat = np.concatenate([data, concat], axis=1)
    def get_train_and_test_size(self):
        return len(self.train_set),len(self.test_set)
    def shuffle(self):
        np.random.shuffle(self.ress)
        self.train_set = self.ress[:,:-self.batch_size]
        self.test_set = self.ress[:,-self.batch_size:]
    def next(self):
        ret = self.train_set[self.counter*self.batch_size:(self.counter+1)*self.batch_size]
        self.counter = (self.counter+1)/self.num_batch
        return ret
    def test_set(self):
        return self.test_set

if __name__ == '__main__':
    bg = BatchGenerator("vol.csv",10,512,32)
    bg.get_data()