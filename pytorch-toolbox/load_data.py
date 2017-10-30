import os
import numpy as np
from .some_script import get_word_from_file_path
from torch.utils.data import Dataset
import torch


class Sample(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label


def load_onefile_data(filepath, word_id):
    fr = open(filepath)
    word = get_word_from_file_path(file_path=filepath)
    dataMat = []
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        float_line = [float(v) for v in curLine]
        dataMat.append(float_line)
    dataMat = np.array(dataMat).transpose(1, 0)
    sample = Sample(data=dataMat, label=word_id[word])
    return sample


def load_dataset(root, word_id, hold_id=-1):
    """
    :param root: data root
    :param word_id : a dict word:id
    :return: a tuple of list, (train_samples_list, dev_samples_list)
    """
    """
    root --> sub-dir --> file
    """
    train_samples = []
    dev_samples = []

    dir_names = sorted([x for x in os.listdir(root)])

    train_dir_names = dir_names[:hold_id]
    if hold_id != -1:
        train_dir_names.extend(dir_names[hold_id + 1:])
    dev_dir_names = [dir_names[hold_id]]

    for train_dir_name in train_dir_names:
        sub_path = os.path.join(root, train_dir_name)
        file_names = sorted(os.listdir(sub_path))
        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            sample = load_onefile_data(filepath=file_path, word_id=word_id)
            train_samples.append(sample)

    for dev_dir_name in dev_dir_names:
        sub_path = os.path.join(root, dev_dir_name)
        file_names = sorted(os.listdir(sub_path))
        for file_name in file_names:
            file_path = os.path.join(sub_path, file_name)
            sample = load_onefile_data(filepath=file_path, word_id=word_id)
            dev_samples.append(sample)

    return train_samples, dev_samples


class SignLangDataset(Dataset):
    def __init__(self, sample_list):
        assert len(sample_list) > 1
        assert isinstance(sample_list[0], Sample)
        self.sample_list = sample_list

    def __getitem__(self, index):
        sample, label = self.sample_list[index].data, self.sample_list[index].label
        sample = torch.FloatTensor(sample)
        return sample, label

    def __len__(self):
        return len(self.sample_list)


class SignLangDataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=True):
        assert isinstance(dataset, SignLangDataset)
        self.batch_size = batch_size
        self.dataset = dataset
        if shuffle:
            indices = iter(torch.randperm(len(self.dataset)).long())
        else:
            indices = iter(range(len(self.dataset)))

        self.indices = indices

    def __iter__(self):
        return self

    def __next__(self):
        idxs = []
        for _ in range(self.batch_size):
            try:
                idxs.append(next(self.indices))
            except:
                if not idxs:
                    # if for statement detect the StopIteration Error
                    # it will not do the next loop
                    raise StopIteration
                break
        batch_data = []
        batch_label = []
        for idx in idxs:
            sample, label = self.dataset[idx]
            batch_data.append(sample)
            batch_label.append(label)
        # 3-D [bs, height, width]
        batch_data = torch.stack(batch_data)
        # 4-D [bs, 1, height, width]
        batch_data = torch.unsqueeze(batch_data, dim=1)

        batch_label = torch.LongTensor(batch_label)
        return batch_data, batch_label

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))


def main():
    root = "/home/fanyang/PycharmProjects/SignLanguage/data/tctodd"
    file_name = "/home/fanyang/PycharmProjects/SignLanguage/data/tctodd/tctodd1/alive-1.tsd"
    mat = load_onefile_data(file_name)
    print(mat.shape)

    pass


if __name__ == '__main__':
    main()
