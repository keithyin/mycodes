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
"""

from nets.models import inception_v3
import torch
from torch.autograd import Variable
from torch import nn
import progressbar
import os
import sys
from data.read_data_pyt import get_loader
from data.read_data_pyt import get_dataset
from tensorboardX import SummaryWriter

Proj_Dir = sys.path[0]

GLOBAL_STEP = 0
EPOCH = 0

BATCH_SIZE = 48


def validation(loader, model, writer):
    model.eval()

    widgets = ["processing: ", progressbar.Percentage(),
               " ", progressbar.ETA(),
               " ", progressbar.FileTransferSpeed(),
               ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(loader)).start()

    # [batch_label, batch_label, ...]
    labels = []
    # [batch_feature, batch_feature, ...]
    features = []
    for i, batch in enumerate(loader):
        bar.update(i)
        batch_data, batch_label = batch

        batch_data = Variable(batch_data, volatile=True).cuda()
        batch_label = Variable(batch_label).cuda()
        logits = model(batch_data)

        labels.append(batch_label)
        features.append(logits)
        if i > 50:
            break

    labels = torch.cat(labels, dim=0).cpu().data

    # all features [50723, 2048]
    #
    features = torch.cat(features, dim=0).cpu().data
    print(features.size())

    labels = [str(v) for v in list(labels.numpy())]
    writer.add_embedding(mat=features, metadata=labels, global_step=0)

    bar.finish()


def main():
    val_dataset = get_dataset(root="/media/fanyang/workspace/DataSet/MARS/bbox_train",
                              txt_file=os.path.join(Proj_Dir, 'data/test.txt'))

    # prepare model
    inception = inception_v3(pretrained=False, aux_logits=False, transform_input=True)

    inception.fc = nn.Linear(in_features=2048, out_features=625)
    inception.load_state_dict(
        torch.load(
            "/home/fanyang/PycharmProjects/PersonReID_CL/ckpt/model-inceptionv3-transform-input.pkl"))

    # inception.optimizer = optim.Adam(params=inception.fc.parameters(), lr=5e-4)

    # set the last layer to None for using the penultimate layer's feature

    inception.fc = None

    inception.cuda()

    # writer = SummaryWriter(log_dir=os.path.join(Proj_Dir, "ckpt/sgd-5e-5"))
    writer_for_embedding = SummaryWriter(log_dir=os.path.join(Proj_Dir, "ckpt/cnn_embedding"))

    val_loader = get_loader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=False)

    validation(loader=val_loader, model=inception, writer=writer_for_embedding)


if __name__ == '__main__':
    main()
