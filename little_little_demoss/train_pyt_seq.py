"""
training-->epoch:10,mean_loss:[ 1.45261502], mean_accuracy:[ 0.65235287]
Adam(params=parameters_need_to_train, lr=1e-3)

training-->epoch:20,mean_loss:[ 0.9952966], mean_accuracy:[ 0.75958961]
optim.Adam(params=parameters_need_to_train, lr=5e-4)


"""

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from torch import cuda
import progressbar
import tools
import os
import sys
from data.read_data_pyt import get_loader
from data.read_data_pyt import get_dataset_seq
from tensorboardX import SummaryWriter
from nets.models import MixModel
from tools import adjust_learning_rate

Proj_Dir = sys.path[0]

GLOBAL_STEP = 0
EPOCH = 0

BATCH_SIZE = 10
SEQ_LEN = 5


def preprocess_batch_data(data, seq_len):
    """
    [batch_size, seq+1, 3, 299, 299] to
    [batch_size, 3, 299, 299] and [batch_size*seq, 3, 299, 299]
    :param data:
    :return:
    """
    batch_size = len(data)
    assert len(data[0]) == (seq_len + 1)
    res_batch_data_shape = (batch_size, data.size(2), data.size(3), data.size(4))
    res_batch_seq_data_shape = (batch_size * seq_len, data.size(2), data.size(3), data.size(4))

    batch_data = []
    batch_seq_data = []
    for i in range(batch_size):
        batch_data.append(data[i][-1])
        batch_seq_data.append(data[i][:-1])
    batch_data = torch.stack(batch_data, dim=0)
    batch_seq_data = torch.cat(batch_seq_data, dim=0)

    assert batch_data.size() == res_batch_data_shape
    assert batch_seq_data.size() == res_batch_seq_data_shape
    return batch_data, batch_seq_data


def train_one_epoch(loader, model, writer):
    global GLOBAL_STEP
    global EPOCH
    model.train()
    total_loss = cuda.FloatTensor([0.])
    total_accu = cuda.FloatTensor([0.])

    widgets = ["processing: ", progressbar.Percentage(),
               " ", progressbar.ETA(),
               " ", progressbar.FileTransferSpeed(),
               ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(loader)).start()

    for i, batch in enumerate(loader):

        bar.update(i)
        # batch_data [batch_size, seq_len+1, 3, 299, 299]
        # the last one of sub sequence is still image
        mix_batch_data, batch_label = batch

        batch_data, batch_seq_data = preprocess_batch_data(mix_batch_data,
                                                           seq_len=SEQ_LEN)

        batch_data = Variable(batch_data).type(torch.FloatTensor).cuda()
        batch_seq_data = Variable(batch_seq_data).type(torch.FloatTensor).cuda()

        batch_label = Variable(batch_label).type(torch.LongTensor).cuda()
        logits = model(batch_data, batch_seq_data)

        loss = model.criterion(logits, batch_label)
        accu = tools.accuracy(logits=logits, targets=batch_label).data
        total_accu.add_(accu)

        total_loss.add_(loss.data)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        if i % 100 == 0:
            writer.add_scalar('train/loss',
                              scalar_value=
                              total_loss.cpu().numpy() / (i + 1), global_step=GLOBAL_STEP // 100)
            writer.add_scalar('train/accu',
                              scalar_value=
                              total_accu.cpu().numpy() / (i + 1), global_step=GLOBAL_STEP // 100)

        GLOBAL_STEP = GLOBAL_STEP + 1

    mean_loss = total_loss.cpu().numpy() / (i + 1)
    mean_accu = total_accu.cpu().numpy() / (i + 1)

    print('')
    print("training-->epoch:{},mean_loss:{}, mean_accuracy:{}".
          format(EPOCH, mean_loss, mean_accu))
    torch.save(model.state_dict(), os.path.join(Proj_Dir, 'ckptseq/model-v1.pkl'))
    bar.finish()
    EPOCH += 1


def validation(loader, model, writer):
    model.eval()
    total_loss = cuda.FloatTensor([0.])
    total_accu = cuda.FloatTensor([0.])

    widgets = ["processing: ", progressbar.Percentage(),
               " ", progressbar.ETA(),
               " ", progressbar.FileTransferSpeed(),
               ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(loader)).start()
    for i, batch in enumerate(loader):
        bar.update(i)
        batch_data, batch_label = batch

        batch_data = Variable(batch_data).cuda()
        batch_label = Variable(batch_label).cuda()
        logits = model(batch_data)

        loss = model.criterion(logits, batch_label)
        accu = tools.accuracy(logits=logits, targets=batch_label).data
        total_accu.add_(accu)

        total_loss.add_(loss.data)

    mean_loss = total_loss.cpu().numpy() / (i + 1)
    mean_accu = total_accu.cpu().numpy() / (i + 1)
    writer.add_scalar('val/loss', scalar_value=mean_loss, global_step=EPOCH)
    writer.add_scalar('val/accu', scalar_value=mean_accu, global_step=EPOCH)

    print('')
    print("validation-->epoch:{},mean_loss:{}, mean_accuracy:{}".
          format(EPOCH, mean_loss, mean_accu))
    bar.finish()


def main():
    # get dataset
    # train_dataset = get_dataset(root="/media/fanyang/workspace/DataSet/MARS/bbox_train",
    #                             txt_file=os.path.join(Proj_Dir, 'data/train.txt'))
    train_dataset = get_dataset_seq(root="/media/fanyang/workspace/DataSet/MARS/bbox_train",
                                    txt_file=os.path.join(Proj_Dir, 'data/id2folder.txt'),
                                    seq_len=SEQ_LEN)
    # prepare model
    model = MixModel(num_classes=625, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
                     transform_input=True)

    # model.load_inception_weights(
    #     "/home/fanyang/PycharmProjects/PersonReID_CL/ckpt/model-inceptionv3-transform-input.pkl")

    model.load_state_dict(
        torch.load("/home/fanyang/PycharmProjects/PersonReID_CL/ckptseq/model-v1.pkl"))

    for param in model.inception.parameters():
        param.requires_grad = False

    # get the parameters to train
    parameters_need_to_train = []
    for param in model.inception_lstm.parameters():
        parameters_need_to_train.append(param)
    for param in model.fc.parameters():
        parameters_need_to_train.append(param)

    model.criterion = nn.CrossEntropyLoss()
    # model.optimizer = optim.Adam(params=parameters_need_to_train, lr=5e-4)
    model.optimizer = optim.SGD(params=parameters_need_to_train, lr=1e-4)
    model.cuda()

    writer = SummaryWriter(log_dir=os.path.join(Proj_Dir, "ckpt/seq-sgd-1e-4"))

    while True:
        train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)

        # print(type(batch_data))
        # print(type(batch_label))
        # exit()
        train_one_epoch(loader=train_loader, model=model, writer=writer)

        adjust_learning_rate(model.optimizer, decay_rate=.9)


if __name__ == '__main__':
    main()
