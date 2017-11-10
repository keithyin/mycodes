import torch
from torch.autograd import Variable


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


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def main():
    from torch.autograd import Variable
    from torch import optim

    opppp = optim.SGD(params=[Variable(torch.FloatTensor(3), requires_grad=True)], lr=0.1)
    print(opppp.state_dict())
    adjust_learning_rate(opppp)
    print(opppp.state_dict())
    pass


if __name__ == '__main__':
    main()
