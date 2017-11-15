import torch
from torch.nn import Module
import torch.nn as nn
from torch import optim
from torch.nn import init
from .padd2d import Padding2D
import math
from torch.autograd import Variable
import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        assert isinstance(x, Variable)
        res = x.view(-1, int(np.prod(x.size()[1:])))
        return res


class QNetwork(Module):
    """
    Q-network, the inputs is [batch_size, 12, 128, 199]
    """

    def __init__(self, num_actions=2):
        super(QNetwork, self).__init__()
        pad_s4_8x8 = Padding2D(kernel_size=8, stride=4)
        pad_s2_4x4 = Padding2D(kernel_size=7, stride=4)
        pad_s1_3x3 = Padding2D(kernel_size=5, stride=4)
        relu = nn.ReLU(inplace=True)

        self.block = nn.Sequential(
            pad_s4_8x8,
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(num_features=32, affine=False),
            relu,
            pad_s2_4x4,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=4),
            nn.BatchNorm2d(num_features=64, affine=False),
            relu,
            pad_s1_3x3,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=4),
            nn.BatchNorm2d(num_features=64, affine=False),
            relu,
            Flatten(),
            nn.Linear(in_features=math.ceil(128 / 64) * math.ceil(199 / 64) * 64, out_features=512),
            nn.BatchNorm1d(num_features=512),
            relu,
            nn.Linear(in_features=512, out_features=num_actions)
        )

        self.optimizer = None

    def forward(self, inputs):
        out = self.block(inputs)
        return out

    def reset_parameters(self, model=None):
        if model is None:
            for layer in self.children():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    init.xavier_normal(layer.weight)
                    init.constant(layer.bias, 0)
            print('the parameters have been initialized.')
        else:
            if not isinstance(model, QNetwork):
                raise ValueError("model must be the same class with this object")
            for param, model_param in zip(self.parameters(), model.parameters()):
                param.data.copy_(model_param.data)
            print('parameters transferred successfully.')

    def get_optimizer(self):
        if self.optimizer is None:
            self.optimizer = optim.RMSprop(self.parameters(), lr=1e-5)
            return self.optimizer
        else:
            return self.optimizer

    def save_model(self, file_path, global_step=None):
        if global_step is None:
            torch.save(self.state_dict(), file_path)
        print('the model has been saved to %s' % file_path)

    def restore_model(self, file_path, global_step=None):
        if global_step is None:
            self.load_state_dict(torch.load(file_path))
        print('the model has been loaded from %s ' % file_path)


def main():
    qnet = QNetwork()
    opt = qnet.get_optimizer()


if __name__ == '__main__':
    main()
