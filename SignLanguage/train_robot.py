from torch import nn
from data.robot_data import load_data_robot, k_fold
from itertools import count
from utils import tools
from torch import optim
from tensorboardX import SummaryWriter
import os
from torch.autograd import Variable
from nets.models import SimpleNN1DCNNRobot
import torch


def normalize_data(train_data, dev_data):
    """
    normalize the data
    :param data: [batch, 1, height, width]
    :return:
    """
    data = torch.cat([train_data, dev_data])
    data = torch.squeeze(data)
    # [batch, height, width]
    mean_data = torch.mean(data, dim=0)
    std_data = torch.std(data, dim=0)

    # [height, width]
    normalized_train_data = (train_data - mean_data) / std_data
    # print(normalized_train_data)
    # exit()
    normalized_dev_data = (dev_data - mean_data) / std_data
    return normalized_train_data, normalized_dev_data


def main():
    root = "/home/fanyang/PycharmProjects/SignLanguage/data/robotfailuer"
    file_name = 'lp1.data.txt'
    file_path = os.path.join(root, file_name)
    dataset = load_data_robot(file_name=file_path)
    print("num data in dataset ", len(dataset))

    for i in range(5):

        train_data, dev_data = k_fold(dataset=dataset, bin_id=i)

        train_batch, train_label = train_data

        dev_batch, dev_label = dev_data

        train_batch, dev_batch = normalize_data(train_batch, dev_batch)

        # model

        model = SimpleNN1DCNNRobot(num_classes=4)
        model.cuda()
        model.criterion = nn.CrossEntropyLoss()

        # using sgd instead of Adam
        model.optimizer = optim.SGD(params=model.parameters(), lr=1e-4)

        train_batch = Variable(train_batch).cuda()
        train_label = Variable(train_label).cuda()

        dev_batch = Variable(dev_batch, volatile=True).cuda()
        dev_label = Variable(dev_label, volatile=True).cuda()

        # print(train_label.size(), train_batch.size())
        # print(dev_label.size(), dev_batch.size())
        # exit()

        # prepare writer
        writer_dir = 'ckpt/robot-1-bin-id-%d' % i
        saver_dir = writer_dir

        writer = SummaryWriter(writer_dir)

        for i in count():
            model.train()
            logits = model(train_batch, writer)

            loss = model.criterion(logits, train_label)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            print("epoch:{}, loss:{}".format(i, loss.cpu().data.numpy()[0]))

            writer.add_scalar('train/loss', loss.cpu().data.numpy(), global_step=i)
            writer.add_scalar('train/accu',
                              tools.accuracy(logits, train_label).cpu().data.numpy(),
                              global_step=i)

            torch.save(model.state_dict(),
                       os.path.join(saver_dir,
                                    'record-step-%d-model.pkl' % i))

            # just save the latest 5 parameters checkpoints
            if i >= 5:
                os.remove(os.path.join(saver_dir,
                                       'record-step-%d-model.pkl' % i))

            tools.adjust_learning_rate(model.optimizer)

            # switch the model to the evaluation mode
            model.eval()
            logits = model(dev_batch)
            loss = model.criterion(logits, dev_label)
            writer.add_scalar('val/loss', loss.cpu().data.numpy(), global_step=i)
            writer.add_scalar('val/accu',
                              tools.accuracy(logits, dev_label).cpu().data.numpy(),
                              global_step=i)


if __name__ == '__main__':
    main()
