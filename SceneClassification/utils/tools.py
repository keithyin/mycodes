"""
the data is balanced
min_width:130, min_height:98, max_width:1000, max_height:1000
"""

import json
import data
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable


def analyze_data(json_file_name):
    f = open(json_file_name)
    dejson = json.load(f)
    labels = []
    for i in range(len(dejson)):
        labels.append(int(dejson[i]['label_id']))
    plt.hist(labels)
    plt.show()
    # the data is balanced


def get_samples(json_file_name):
    assert isinstance(json_file_name, str)
    splited_path = json_file_name.split('/')
    splited_path.pop(-1)
    splited_path.pop(0)
    splited_path.append('scene_train_images_20170904')
    tmp = [v + '/' for v in splited_path]
    img_data_dir = '/' + ''.join(tmp)

    f = open(json_file_name)
    dejson = json.load(f)
    count = [0] * 80

    train_sample_list = []
    val_sample_list = []

    for i in range(len(dejson)):
        label = int(dejson[i]['label_id'])
        img_name = dejson[i]['image_id']
        img_path = os.path.join(img_data_dir, img_name)
        if count[label] < 125:
            count[label] = count[label] + 1
            sample = data.Sample(img_path=img_path, label=label)
            val_sample_list.append(sample)
        else:
            sample = data.Sample(img_path=img_path, label=label)
            train_sample_list.append(sample)
    return train_sample_list, val_sample_list


def height_width(img_dir):
    file_names = os.listdir(img_dir)
    file_paths = [os.path.join(img_dir, img_name) for img_name in file_names]
    max_width = -np.inf
    max_height = -np.inf
    min_width = np.inf
    min_height = np.inf
    for file_path in file_paths:
        img = Image.open(file_path)
        width = img.width
        height = img.height
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        min_width = min(min_width, width)
        min_height = min(min_height, height)
    print("min_width:%d, min_height:%d, max_width:%d, max_height:%d" % (min_width, min_height,
                                                                        max_width, max_height))


def main():
    file_name = \
        '/media/fanyang/workspace/DataSet/ai_challenger_scene_train_20170904/' \
        'scene_train_annotations_20170904.json'
    img_dir = '/media/fanyang/workspace/DataSet/ai_challenger_scene_train_20170904/' \
              'scene_train_images_20170904'
    tr_list, val_list = get_samples(file_name)
    print(len(tr_list), len(val_list))


def accuracy(logits, targets):
    """
    cal the accuracy of the predicted result
    :param logits: Variable [batch_size, num_classes]
    :param targets: Variable [batch_size]
    :return: Variable scalar
    """
    assert isinstance(logits, Variable)
    val, idx = logits.max(dim=1)
    eql = (idx == targets)
    eql = eql.type(torch.cuda.FloatTensor)
    res = torch.mean(eql)
    return res


if __name__ == '__main__':
    main()
