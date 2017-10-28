import progressbar
from torch.autograd import Variable
import torch
import os
import sys
from torch import cuda

Proj_Dir = sys.path[0]


def train_one_epoch(loader, model, writer, record_n_times_per_epoch=10):
    glos = globals()
    # initialize the record step
    if '_record_step' not in glos:
        glos['_record_step'] = 0
    if '_record_epoch_step' not in glos:
        glos['_record_epoch_step'] = 0

    record_step = glos['_record_step']
    record_epoch_step = glos['_record_epoch_step']

    if len(loader) < record_n_times_per_epoch:
        raise ValueError("record_n_times_per_epoch {} is larger than loader size {}"
                         .format(record_n_times_per_epoch, len(loader)))

    record_interval = len(loader) // record_n_times_per_epoch

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
        batch_data, batch_label = batch

        batch_data = Variable(batch_data).type(torch.FloatTensor).cuda()
        batch_label = Variable(batch_label).type(torch.LongTensor).cuda()
        logits = model(batch_data)

        loss = model.criterion(logits, batch_label)

        accu = accuracy(logits=logits, targets=batch_label).data
        total_accu.add_(accu)

        total_loss.add_(loss.data)

        # using two criterion to calculate the gradient
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        if (i + 1) % record_interval == 0:
            writer.add_scalar('train/loss',
                              scalar_value=total_loss.cpu().numpy() / (i + 1),
                              global_step=record_step)
            writer.add_scalar('train/accu',
                              scalar_value=total_accu.cpu().numpy() / (i + 1),
                              global_step=record_step)
            glos['_record_step'] = glos['_record_step'] + 1
            if i != 0:
                torch.save(model.state_dict(),
                           os.path.join(Proj_Dir,
                                        'ckpt/model.pkl'))

    mean_loss = total_loss.cpu().numpy() / (i + 1)
    mean_accu = total_accu.cpu().numpy() / (i + 1)

    print('')
    print("training-->epoch:{},mean_loss:{}, mean_accuracy:{}".
          format(record_epoch_step, mean_loss, mean_accu))

    bar.finish()
    glos['_record_epoch_step'] = glos['_record_epoch_step'] + 1


def validation(loader, model, writer):
    glos = globals()
    # initialize the record step
    if '_record_epoch_step' not in glos:
        glos['_record_epoch_step'] = 0

    record_epoch_step = glos['_record_epoch_step']

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

        batch_data = Variable(batch_data, volatile=True).cuda()
        batch_label = Variable(batch_label, volatile=True).cuda()
        logits = model(batch_data)

        loss = model.criterion(logits, batch_label)
        accu = accuracy(logits=logits, targets=batch_label).data
        total_accu.add_(accu)

        total_loss.add_(loss.data)
    #
    mean_loss = total_loss.cpu().numpy() / (i + 1)
    mean_accu = total_accu.cpu().numpy() / (i + 1)
    writer.add_scalar('val/loss', scalar_value=mean_loss, global_step=record_epoch_step)
    writer.add_scalar('val/accu', scalar_value=mean_accu, global_step=record_epoch_step)
    #
    print('')
    print("validation-->epoch:{},mean_loss:{}, mean_accuracy:{}".
          format(record_epoch_step, mean_loss, mean_accu))

    bar.finish()


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
