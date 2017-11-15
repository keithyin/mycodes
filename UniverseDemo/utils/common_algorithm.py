import numpy as np
from torch.autograd import Variable
import torch


def epsilon_greedy(logits, num_actions, epsilon=1.):
    """
    perform epsilon greedy action selection algorithm
    :param logits: Variable, shape [batch_size, num_actions]
    :param num_actions: int
    :param epsilon: [0., 1.], the probability of choosing random action,
    when epsilon=0, this is the greedy algorithm
    :return:the selected action
    when will use this function:
    behavior policy: choose action based on current state
    """
    assert isinstance(logits, Variable)
    if np.random.uniform() < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = logits.max(dim=1)[1].cpu().data.numpy()[0]
    return action


def one_hot(ids, out_tensor):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    if not isinstance(ids, (list, np.ndarray)):
        raise ValueError("ids must be 1-D list or array")
    if isinstance(ids, np.ndarray):
        ids = ids.astype(dtype=np.int64)
    ids = torch.LongTensor(ids).view(-1, 1)
    out_tensor.zero_()
    out_tensor.scatter_(dim=1, index=ids, value=1.)
    # out_tensor.scatter_(1, ids, 1.0)
