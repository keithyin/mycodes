from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
from collections import OrderedDict


class Img(object):
    def __init__(self, img_path, label):
        self.img_path = img_path
        self.label = label


class Person(object):
    def __init__(self, folder, id):
        self.folder = folder
        self.id = id


def get_seqlist_from_folder(folder):
    seqlist = []
    file_names = os.listdir(folder)
    file_names = sorted(file_names)
    seq_tracker = []
    sequence = None
    for file_name in file_names:
        camera = str(file_name[4:11])
        abs_file_name = os.path.join(folder, file_name)
        if camera not in seq_tracker:
            if sequence is not None:
                seqlist.append(sequence)
            seq_tracker.append(camera)
            sequence = []
        sequence.append(abs_file_name)
    seqlist.append(sequence)
    return seqlist


def parse_txt_file(file_name, data_dir):
    """
    using txt file and data root dir to generate a list of Img object using to build records
    :param file_name: train_txt file name or test_txt file name
    :param data_dir: data root dir, using to generate the absolute path of the image
    :return: a list of Img objects
    """
    imgs = []
    with open(file_name, mode="r") as file:
        try:
            for line in file:
                img_label = line.split(" ")

                img_path = os.path.join(data_dir, img_label[0])
                label = int(img_label[1])
                imgs.append(Img(img_path, label))
        except KeyboardInterrupt as e:
            file.close()
    return imgs


def parse_txt_file_to_person_list(file_name, data_dir):
    """
    using txt file and data root dir to generate a list of Img object using to build records
    :param file_name: train_txt file name or test_txt file name
    :param data_dir: data root dir, using to generate the absolute path of the image
    :return: a list of Img objects
    """
    persons = []
    with open(file_name, mode="r") as file:
        try:
            for line in file:
                person = line.split(" ")

                person_folder = os.path.join(data_dir, person[0])
                label = int(person[1])
                persons.append(Person(person_folder, label))
        except KeyboardInterrupt as e:
            file.close()
    return persons


class CustomDataset(Dataset):
    def __init__(self, root, txt_file, train=True, transform=None):
        self.train = train
        self.data = parse_txt_file(txt_file, data_dir=root)

        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data[index].img_path, self.data[index].label
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)


class CustomDatasetCNNTest(Dataset):
    pass


class CustomDatasetSeq(Dataset):
    def __init__(self, root, txt_file, seq_len, train=True, transform=None):
        """
        Dataset for Seq classification
        :param root: data root
        :param txt_file: id2folder, giving every folder in data root a label
        :param seq_len:
        :param train:
        :param transform:
        """
        self.seq_len = seq_len
        self.train = train
        # using it to get the person id and corresponded file folder
        self.person_list = parse_txt_file_to_person_list(txt_file, root)
        # using it to store the sequence list corresponding with the person
        self.personid_seqlist = {}
        self.datasets = []

        self.num_person = len(self.person_list)

        self.transform = transform

        self.__fill_personid_seqlist()
        self.__fill_datasets()

    def __getitem__(self, index):
        seq, label = self.datasets[index]

        # pick random sub sequence
        end_point = np.random.randint(self.seq_len, len(seq) + 1)
        begin_point = end_point - self.seq_len
        sub_seq = seq[begin_point:end_point]

        # pick random image
        img_idx = np.random.randint(len(seq))
        img_path = seq[img_idx]

        sub_seq.append(img_path)
        imgs = []

        for file in sub_seq:
            img = Image.open(file)
            img = self.transform(img)
            imgs.append(img)
        # imgs [seq_len+1, 3, 299, 299]
        # the last one is the still image
        imgs = torch.stack(imgs, dim=0)

        return imgs, label

    def __fill_personid_seqlist(self):
        # get the person
        # using person.id as key, seqlist as value
        for person in self.person_list:
            self.personid_seqlist[person.id] = get_seqlist_from_folder(folder=person.folder)

    def __fill_datasets(self):
        for label, seqlist in self.personid_seqlist.items():
            for seq in seqlist:
                self.datasets.append([seq, label])

    def max_min_seq_len(self):
        """
        max:  900.0  min: 5.0
        """
        max_val = -np.inf
        min_val = np.inf
        for seq, label in self.datasets:
            max_val = np.maximum(max_val, len(seq))
            min_val = np.minimum(min_val, len(seq))
        print("max: ", max_val, " min:", min_val)

    def __len__(self):
        return len(self.datasets)


class CustomDatasetVal(Dataset):
    def __init__(self, root, txt_file, seq_len, train=True, transform=None, is_query=True):
        """
        Dataset for Seq classification
        :param root: data root
        :param txt_file: id2folder, giving every folder in data root a label
        :param seq_len:
        :param train:
        :param transform:
        """
        self.is_query = is_query
        self.seq_len = seq_len
        self.train = train
        # using it to get the person id and corresponded file folder
        # store the Person obj in the person_list
        self.person_list = parse_txt_file_to_person_list(txt_file, root)

        # using it to store the sequence list corresponding with the person
        self.personid_seqlist = OrderedDict()

        self.gallery_datasets = []
        self.quries = []

        self.num_person = len(self.person_list)

        self.transform = transform

        self.bad_id = 635

        self.__fill_personid_seqlist()

        self.__fill_datasets()

    def __getitem__(self, index):
        if self.is_query:
            datasets = self.quries
        else:
            datasets = self.gallery_datasets
        seq, label = datasets[index]

        # pick random sub sequence
        end_point = np.random.randint(self.seq_len, len(seq) + 1)
        begin_point = end_point - self.seq_len
        sub_seq = seq[begin_point:end_point]

        # pick random image
        img_idx = np.random.randint(len(seq))
        img_path = seq[img_idx]

        sub_seq.append(img_path)
        imgs = []

        for file in sub_seq:
            img = Image.open(file)
            img = self.transform(img)
            imgs.append(img)
        # imgs [seq_len+1, 3, 299, 299]
        # the last one is the still image
        imgs = torch.stack(imgs, dim=0)

        return imgs, label

    def __fill_personid_seqlist(self):
        # get the person
        # using person.id as key, seqlist as value
        for person in self.person_list:
            self.personid_seqlist[person.id] = get_seqlist_from_folder(folder=person.folder)

    def __fill_datasets(self):
        for label, seqlist in self.personid_seqlist.items():
            flag = True
            for seq in seqlist:
                if flag and label != self.bad_id:
                    self.quries.append([seq, label])
                    flag = False
                else:
                    self.gallery_datasets.append([seq, label])

    def max_min_seq_len(self):
        """
        max:  900.0  min: 5.0
        """
        max_val = -np.inf
        min_val = np.inf
        if self.is_query:
            datasets = self.quries
        else:
            datasets = self.gallery_datasets
        for seq, label in datasets:
            max_val = np.maximum(max_val, len(seq))
            min_val = np.minimum(min_val, len(seq))
        print("max: ", max_val, " min:", min_val)

    def get_query_test_count(self):
        res = []
        for i, val in self.personid_seqlist.items():
            res.append(len(val))

        del res[-1]
        return torch.LongTensor(res)

    def __len__(self):
        if self.is_query:
            return len(self.quries)
        else:
            return len(self.gallery_datasets)


def main():
    # ds = CustomDatasetSeq(root="/media/fanyang/workspace/DataSet/MARS/bbox_train",
    #                       txt_file="/home/fanyang/PycharmProjects/PersonReID_CL/data/id2folder.txt",
    #                       seq_len=10)
    # ds.max_min_seq_len()

    dsval = CustomDatasetVal(root="/media/fanyang/workspace/DataSet/MARS/bbox_test",
                             txt_file="/home/fanyang/PycharmProjects/PersonReID_CL/data/id2folderVal.txt",
                             seq_len=5)
    print(len(dsval.quries))


if __name__ == '__main__':
    main()
