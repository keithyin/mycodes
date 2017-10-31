from data.load_data import load_dataset, SignLangDataset, SignLangDataLoader
from data.some_script import id_2_sign
from nets.models import SimpleNN1DCNN
import visdom
import numpy as np
from torch.autograd import Variable
import torch
from torch import optim
from itertools import count
from torch.nn import functional as F
from utils.tools import adjust_learning_rate
from torch import nn


def get_min_max_from_dataset(dataset):
    num_sample = len(dataset)
    min_val = np.inf
    max_val = -np.inf
    for i in range(num_sample):
        data = dataset[i][0]
        min_val = np.minimum(np.min(data.numpy()), min_val)
        max_val = np.maximum(np.max(data.numpy()), max_val)
    return min_val, max_val


def main():
    root = "/home/fanyang/PycharmProjects/SignLanguage/data/tctodd"
    id_sign, sign_id = id_2_sign(root)

    train_samples, dev_samples = load_dataset(root, sign_id, hold_id=8)

    train_dataset = SignLangDataset(sample_list=train_samples)
    dev_dateset = SignLangDataset(sample_list=dev_samples)

    # model

    model = SimpleNN1DCNN()
    model.load_state_dict(
        torch.load('/home/fanyang/PycharmProjects/SignLanguage/ckpt/model.pkl'))
    model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    vis = visdom.Visdom()
    min_val, max_val = get_min_max_from_dataset(train_dataset)
    # min: -0.4535120129585266, max: 1.0
    print("min:{}, max:{}".format(min_val, max_val))
    # exit()
    vis.heatmap(X=train_dataset[0][0], opts={'xmin': min_val, 'xmax': max_val})

    # [22, 90] label :0
    inputs = Variable(torch.randn(1, 1, 22, 90), requires_grad=True).cuda()
    nn.init.normal(inputs, std=.001)

    inputs._grad_fn = None

    model.optimizer = optim.SGD(params=[inputs], lr=1e-4)

    for i in count():
        logits = model(inputs)
        logits = torch.squeeze(logits)
        model.optimizer.zero_grad()
        logits[0].backward(torch.FloatTensor([-1.]).cuda())
        model.optimizer.step()
        if (i + 1) % 50000 == 0:
            vis.heatmap(X=torch.squeeze(inputs).cpu().data,
                        opts={'xmin': min_val, 'xmax': max_val,
                              'title': '%d-step' % (i + 1)})
            print("step:%d" % (i + 1), "prob:%.7f" %
                  F.softmax(logits)[0].cpu().data.numpy()[0])
            adjust_learning_rate(model.optimizer)


if __name__ == '__main__':
    main()
