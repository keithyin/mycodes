import torch
from torch import nn
from torch.nn import functional as F


class SimpleNN1DCNN(nn.Module):
    def __init__(self, num_classes=96):
        super(SimpleNN1DCNN, self).__init__()
        self.num_classes = num_classes

        # [bs, 1, 22, W]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(22, 3),
                               padding=(0, 1))
        # [bs, 64, 1, W]
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        # [bs, 64, 1, W/2]
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3),
                               padding=(0, 1))

        # [bs, 128, 1, W/2]
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        # [bs, 128, 1, W/4]
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3),
                               padding=(0, 1))

        # [bs, 256, 1, W/4]
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))

        # # [bs, 256, 1, W/8]
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=self.num_classes,
                               kernel_size=(1, 3),
                               padding=(0, 1))

    def forward(self, x):
        """
        neural network's forward process
        :param x: [bs, 1, 22, W]
        :return: [bs, num_classes]
        """
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        # x [bs, num_classes, 1, ?]
        x = self.conv4(x)

        x = torch.squeeze(x, dim=2)

        x = torch.mean(x, dim=-1)
        return x
