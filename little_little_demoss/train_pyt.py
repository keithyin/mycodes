"""
training-->epoch:2,mean_loss:[ 1.89013493], mean_accuracy:[ 0.57777607]
validation-->epoch:2,mean_loss:[ 0.96589673], mean_accuracy:[ 0.77282602]
Adam:LastLayer [lr=1e-3]

training-->epoch:42,mean_loss:[ 1.24983406], mean_accuracy:[ 0.69711345]
validation-->epoch:42,mean_loss:[ 0.61198413], mean_accuracy:[ 0.86016458]
Adam:ALLLayer [lr=5e-4] with learning rate decay

training-->epoch:8,mean_loss:[ 0.31749031], mean_accuracy:[ 0.92480028]
validation-->epoch:9,mean_loss:[ 0.17722692], mean_accuracy:[ 0.96573216]
SGD:ALLLayer [lr=5e-5] with learning rate decay



with CenterLoss
training-->epoch:4,mean_loss:[ 0.26399836], mean_accuracy:[ 0.96862757]
validation-->epoch:5,mean_loss:[ 0.11369684], mean_accuracy:[ 0.98462802]

"""
from torch import nn
from torch import optim
import os
import sys
from data.read_data_pyt import get_loader
from data.read_data_pyt import get_dataset
from tensorboardX import SummaryWriter
from utils.tools import adjust_learning_rate
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
    inceptionv2 = Inceptionv2()

    inceptionv2.criterion = nn.CrossEntropyLoss()

    inceptionv2.cuda()

    # inceptionv2.load_state_dict(
    #     torch.load(
    #         "/home/fanyang/PycharmProjects/PersonReID_CL/ckpt/model-inceptionv3-transform-input.pkl"))

    inceptionv2.optimizer = optim.Adam(params=inceptionv2.parameters(), lr=1e-3)
    # inception.optimizer = optim.SGD(params=inception.parameters(), lr=5e-4)


    # for record the train process
    writer_dir = "ckpt/no-cl-model-inceptionv2"
    saver_dir = writer_dir
    writer = SummaryWriter(log_dir=os.path.join(Proj_Dir, writer_dir))

    routine = Routine(model=inceptionv2, saver_dir=saver_dir, writer=writer)

    while True:
        train_loader = get_loader(dataset=train_dataset, batch_size=BATCH_SIZE)
        val_loader = get_loader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                drop_last=False)

        routine.train_one_epoch(loader=train_loader, record_n_times_per_epoch=400)

        # adjust the learning per epoch
        adjust_learning_rate(inceptionv2.optimizer)

        routine.validation(loader=val_loader)


if __name__ == '__main__':
    main()
