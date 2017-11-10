"""
No-center-loss:
Last Layer Adam(params=resnet.fc.parameters(), lr=1e-3)
training-->epoch:1,mean_loss:[ 0.60688722], mean_accuracy:[ 0.85004115]
validation-->epoch:1,mean_loss:[ 0.58991349], mean_accuracy:[ 0.85250789]

ALL layer: SGD(params=resnet.parameters(), lr=5e-4)
training-->epoch:6,mean_loss:[ 0.02466226], mean_accuracy:[ 0.99575287]
validation-->epoch:6,mean_loss:[ 0.06963665], mean_accuracy:[ 0.98457205]

Center Loss:
ALL layer: SGD(params=resnet.parameters(), lr=5e-4)

"""

from nets.models import resnet50
import torch
from torch import nn
from torch import optim
import os
import sys
from data.read_data_pyt import get_loader
from data.read_data_pyt import get_dataset
from tensorboardX import SummaryWriter
from utils.tools import adjust_learning_rate
from utils import tools
from nets.losses import CenterLoss
from nets.losses import ImprovedCenterLoss
from utils.routine import Routine2Criteion
from utils.routine import Routine
from nets.gv2model import Inceptionv2

Proj_Dir = sys.path[0]

BATCH_SIZE = 64


def main():
    # get dataset
    train_dataset = get_dataset(root="/media/fanyang/workspace/DataSet/MARS/bbox_train",
                                txt_file=os.path.join(Proj_Dir, 'data/train.txt'))
    val_dataset = get_dataset(root="/media/fanyang/workspace/DataSet/MARS/bbox_train",
                              txt_file=os.path.join(Proj_Dir, 'data/test.txt'))

    # prepare model
    model = Inceptionv2()

    # two criterion
    model.criterion = nn.CrossEntropyLoss()
    model.criterion2 = CenterLoss(feature_len=1024, num_classes=625)

    # model.load_state_dict(
    #     torch.load(
    #         "/home/fanyang/PycharmProjects/PersonReID_CL/ckpt/no-cl-model-resnet50/last-layer-finetuned-model.pkl"))

    model.cuda()
    model.optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    # model.optimizer = optim.SGD(params=model.parameters(), lr=1e-3)

    # for record the train process
    writer_dir = "ckpt/cl-model-inceptionv2"
    saver_dir = writer_dir
    writer = SummaryWriter(log_dir=os.path.join(Proj_Dir, writer_dir))

    routine = Routine2Criteion(model=model, saver_dir=saver_dir, writer=writer)

    while True:
        train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
        val_loader = get_loader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                drop_last=False)

        # print(type(batch_data))
        # print(type(batch_label))
        # exit()

        routine.train_one_epoch(loader=train_loader, record_n_times_per_epoch=400)

        # adjust the learning per epoch
        adjust_learning_rate(model.optimizer)

        routine.validation(loader=val_loader)


if __name__ == '__main__':
    main()
