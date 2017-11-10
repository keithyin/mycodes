from torch import nn
import torch
from torch.nn import functional as F
import math


class Inceptionv2(nn.Module):
    def __init__(self, num_classes=625):
        super(Inceptionv2, self).__init__()

        pad_s2_7x7 = Padding2D(kernel_size=7, stride=2, padding='SAME')
        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')
        pad_s2_3x3 = Padding2D(kernel_size=3, stride=2, padding='SAME')

        self.base = nn.Sequential(
            pad_s2_7x7,
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2),  # 64, 112, 112
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),  # 64, 56, 56

            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=64, affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=64, out_channels=192,
                      stride=1, kernel_size=3),  # 192, 56, 56
            nn.BatchNorm2d(num_features=192, affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.MaxPool2d(kernel_size=3,
                         stride=2)  # 192, 28, 28
        )

        self.other_block = nn.Sequential(
            Mix3b(), Mix3c(), Mix4a(), Mix4b(), Mix4c(), Mix4d(), Mix4e(),
            Mix5a(), Mix5b(), Mix5c()
        )

        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        res = self.other_block(self.base(x))

        # the last feature layer no relu
        # [1024, 7, 7]
        res = torch.mean(res, dim=-1)
        res = torch.mean(res, dim=-1)

        self.feature = res

        if self.classifier is None:
            return res

        res = self.classifier(res)
        return res


class Mix3b(nn.Module):
    def __init__(self):
        super(Mix3b, self).__init__()
        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')
        in_channels = 192

        branch_0_hyper_param = [64]
        branch_1_hyper_param = [64, 64]
        branch_2_hyper_param = [64, 96, 96]
        branch_3_hyper_param = [32]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())

        # torch.Size([10, 256, 28, 28])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mix3c(nn.Module):
    def __init__(self):
        super(Mix3c, self).__init__()
        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        in_channels = 256

        branch_0_hyper_param = [64]
        branch_1_hyper_param = [64, 96]
        branch_2_hyper_param = [64, 96, 96]
        branch_3_hyper_param = [64]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 320, 28, 28])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mix4a(nn.Module):
    def __init__(self):
        super(Mix4a, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')
        pad_s2_3x3 = Padding2D(kernel_size=3, stride=2, padding='SAME')

        in_channels = 320

        branch_0_hyper_param = [128, 160]
        branch_1_hyper_param = [64, 96, 96]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.Conv2d(in_channels=branch_0_hyper_param[0],
                      out_channels=branch_0_hyper_param[1], kernel_size=3,
                      stride=2),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[1],
                      out_channels=branch_1_hyper_param[2], kernel_size=3,
                      stride=2),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            pad_s2_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)

        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())

        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2], dim=1)
        return res


class Mix4b(nn.Module):
    def __init__(self):
        super(Mix4b, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        in_channels = 576

        branch_0_hyper_param = [224]
        branch_1_hyper_param = [64, 96]
        branch_2_hyper_param = [96, 128, 128]
        branch_3_hyper_param = [128]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mix4c(nn.Module):
    def __init__(self):
        super(Mix4c, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        in_channels = 576

        branch_0_hyper_param = [192]
        branch_1_hyper_param = [96, 128]
        branch_2_hyper_param = [96, 128, 128]
        branch_3_hyper_param = [128]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mix4d(nn.Module):
    def __init__(self):
        super(Mix4d, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        in_channels = 576

        branch_0_hyper_param = [160]
        branch_1_hyper_param = [128, 160]
        branch_2_hyper_param = [128, 160, 160]
        branch_3_hyper_param = [96]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mix4e(nn.Module):
    def __init__(self):
        super(Mix4e, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        in_channels = 576

        branch_0_hyper_param = [96]
        branch_1_hyper_param = [128, 192]
        branch_2_hyper_param = [160, 192, 192]
        branch_3_hyper_param = [96]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mix5a(nn.Module):
    def __init__(self):
        super(Mix5a, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')
        pad_s2_3x3 = Padding2D(kernel_size=3, stride=2, padding='SAME')

        in_channels = 576

        branch_0_hyper_param = [128, 192]
        branch_1_hyper_param = [192, 256, 256]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.Conv2d(in_channels=branch_0_hyper_param[0],
                      out_channels=branch_0_hyper_param[1], kernel_size=3,
                      stride=2),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[1],
                      out_channels=branch_1_hyper_param[2], kernel_size=3,
                      stride=2),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            pad_s2_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)

        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())

        # torch.Size([10, 1024, 7, 7])
        res = torch.cat([br_0, br_1, br_2], dim=1)
        return res


class Mix5b(nn.Module):
    def __init__(self):
        super(Mix5b, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        in_channels = 1024

        branch_0_hyper_param = [352]
        branch_1_hyper_param = [192, 320]
        branch_2_hyper_param = [160, 224, 224]
        branch_3_hyper_param = [128]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mix5c(nn.Module):
    def __init__(self):
        super(Mix5c, self).__init__()

        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        in_channels = 1024

        branch_0_hyper_param = [352]
        branch_1_hyper_param = [192, 320]
        branch_2_hyper_param = [192, 224, 224]
        branch_3_hyper_param = [128]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            # nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            # nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            # nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            # nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Mixa(nn.Module):
    def __init__(self, in_channels, branch_hyper_params):
        """
        to generate object of mix4a and mix5a
        :param in_channels: int
        :param branch_hyper_params: list of list
        """
        super(Mixa, self).__init__()
        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')
        pad_s2_3x3 = Padding2D(kernel_size=3, stride=2, padding='SAME')

        branch_0_hyper_param = branch_hyper_params[0]
        branch_1_hyper_param = branch_hyper_params[1]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.Conv2d(in_channels=branch_0_hyper_param[0],
                      out_channels=branch_0_hyper_param[1], kernel_size=3,
                      stride=2),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s2_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[1],
                      out_channels=branch_1_hyper_param[2], kernel_size=3,
                      stride=2),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            pad_s2_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)

        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())

        # torch.Size([10, 1024, 7, 7])
        res = torch.cat([br_0, br_1, br_2], dim=1)
        return res


class Mixbcde(nn.Module):
    def __init__(self, in_channels, branch_details):
        """
        for generate object of mix3b, mix3c, mix4b, ..., mix5b...
        :param in_channels: int
        :param branch_details: list of list
        """
        super(Mixbcde, self).__init__()
        pad_s1_3x3 = Padding2D(kernel_size=3, stride=1,
                               padding='SAME')

        branch_0_hyper_param = branch_details[0]
        branch_1_hyper_param = branch_details[1]
        branch_2_hyper_param = branch_details[2]
        branch_3_hyper_param = branch_details[3]

        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_0_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_0_hyper_param[0], affine=False),
            nn.ReLU()
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_1_hyper_param[0],
                      stride=1,
                      kernel_size=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_1_hyper_param[0],
                      out_channels=branch_1_hyper_param[1], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_1_hyper_param[1], affine=False),
            nn.ReLU()
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=branch_2_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[0], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[0],
                      out_channels=branch_2_hyper_param[1],
                      kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[1], affine=False),
            nn.ReLU(),
            pad_s1_3x3,
            nn.Conv2d(in_channels=branch_2_hyper_param[1],
                      out_channels=branch_2_hyper_param[2], kernel_size=3,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_2_hyper_param[2], affine=False),
            nn.ReLU()
        )
        self.branch_3 = nn.Sequential(
            pad_s1_3x3,
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=branch_3_hyper_param[0],
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm2d(num_features=branch_3_hyper_param[0], affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        br_0 = self.branch_0(x)
        br_1 = self.branch_1(x)
        br_2 = self.branch_2(x)
        br_3 = self.branch_3(x)
        # print('br_0', br_0.size())
        # print('br_1', br_1.size())
        # print('br_2', br_2.size())
        # print('br_3', br_3.size())
        # torch.Size([10, 576, 14, 14])
        res = torch.cat([br_0, br_1, br_2, br_3], dim=1)
        return res


class Padding2D(nn.Module):
    """
    if one wants the the tensorflow's 'SAME' or 'VALID' pad, this Module will help you
    to achieve it. one doesn't need to calculate the Conv2D's padding value, just
    leave it 0!!

    stride : int or tuple, (sH,sW)
    kernel : int or tuple, (kH, kW)
    padding : 'SAME' or 'VALID'

    pad1 = Padding2D(kernel_size=3, stride=2)
    conv1 = Con2d(3, 64, kernel_size=3, stride=2)

    inputs = Variable(torch.Tensor(3,3,224,224)
    res = conv1(pad1(inputs))

    this is the same with Conv2D(....., pad='SAME')

    """

    def __init__(self, kernel_size, stride, padding='SAME'):
        super(Padding2D, self).__init__()
        assert isinstance(stride, (int, tuple))
        assert isinstance(kernel_size, (int, tuple))

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.kernel = kernel_size

        if not (padding == 'SAME' or padding == 'VALID'):
            raise ValueError("padding must be 'SAME' or 'VALID'")
        self.padding = padding

    def forward(self, inputs):

        stride_height, stride_width = self.stride
        kernel_height, kernel_width = self.kernel
        _, _, height, width = inputs.size()

        if self.padding == 'VALID':
            # if using valid padding, just return the inputs
            # this don't need to pad the inputs
            return inputs

        # if padding = 'SAME', we need to pad the feature map
        # now we are going to compute how to pad
        out_height = math.ceil(height / stride_height)
        out_width = math.ceil(width / stride_width)

        wanted_inputs_height = (out_height - 1) * stride_height + kernel_height
        wanted_inputs_width = (out_width - 1) * stride_width + kernel_width

        pad_height = wanted_inputs_height - height
        pad_width = wanted_inputs_width - width

        pad_height_top = pad_height // 2 + pad_height % 2
        pad_height_down = pad_height - pad_height_top

        pad_width_left = pad_width // 2 + pad_width % 2
        pad_width_right = pad_width - pad_width_left

        pad = (pad_width_left, pad_width_right,
               pad_height_top, pad_height_down)

        res = F.pad(inputs, pad=pad, mode='constant', value=0.)
        return res


def main():
    from torch.autograd import Variable
    net = Inceptionv2().cuda()
    inputs = Variable(torch.randn(10, 3, 224, 224).cuda())
    print(net(inputs).size())


if __name__ == '__main__':
    main()
