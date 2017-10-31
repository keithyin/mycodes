import torch
from torch.autograd import Variable


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def accuracy(logits, targets):
    """
    cal the accuracy of the predicted result
    :param logits: Variable [batch_size, num_classes]
    :param targets: Variable [batch_size]
    :return: Variable scalar
    """
    assert isinstance(logits, Variable)
    assert len(logits) == len(targets), "the number of logits and targets must be the same," \
                                        "but you got, " \
                                        "num logits:{}, num targets:{}".format(
        len(logits), len(targets))

    val, idx = logits.max(dim=1)
    eql = (idx == targets)
    eql = eql.type_as(logits)
    res = torch.mean(eql)
    return res
