import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import math
from numbers import Number
from collections import Sequence


class Conv2DWrapper(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="SAME", dilation=1, groups=1,
                 bias=True):

        super(Conv2DWrapper, self).__init__(in_channels, out_channels, kernel_size, stride,
                                            padding=0, dilation=dilation, groups=groups, bias=bias)

        if not (padding == "SAME" or padding == "VALID"):
            raise ValueError("padding is 'SAME' or 'VALID' ")
        self.padding_ = padding
        self.pad_func = Padding2D(kernel_size, stride)

    def forward(self, input):
        if self.padding_ == "SAME":
            input = self.pad_func(input)
        res = super(Conv2DWrapper, self).forward(input=input)
        return res


class ConvTransposed2DWrapper(nn.ConvTranspose2d):
    """
    ConvTransposed is how we compute the gradients of Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding="SAME", stride=1, groups=1, bias=True,
                 dilation=1):
        super(ConvTransposed2DWrapper, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                                      padding=0, output_padding=0, groups=groups, bias=bias,
                                                      dilation=dilation)
        if not (padding == "SAME" or padding == "VALID"):
            raise ValueError("padding is 'SAME' or 'VALID' ")
        # this is use to specify the padding method used in the convolution procedure
        self.padding_ = padding

        if isinstance(kernel_size, Number):
            self.kernel_size_ = np.array([kernel_size, kernel_size])
        elif isinstance(kernel_size, Sequence):
            self.kernel_size_ = np.array(kernel_size)
        else:
            raise ValueError("kernel_size must be Number or Sequence")

        if isinstance(stride, Number):
            self.stride_ = np.array([stride, stride])
        elif isinstance(stride, Sequence):
            self.stride_ = np.array(stride)
        else:
            raise ValueError("stride must be Number or Sequence")

    def _check_shape(self, in_shape, out_shape):
        assert isinstance(in_shape, Sequence)
        assert isinstance(out_shape, Sequence)
        in_shape = np.array(in_shape)
        out_shape = np.array(out_shape)

        pad_size = Padding2D.pad_size(out_shape, kernel_size=self.kernel_size_, stride=self.stride_)
        # this is the max shape
        deconved_shape = self.stride_ * (in_shape - 1) + self.kernel_size_

        # this is the min shape out_shape in  [min, max)
        min_shape = deconved_shape - self.stride
        if not np.sum(min_shape <= out_shape) > 1.5:
            raise ValueError("out_shape is illegal")

        if np.sum(out_shape>=deconved_shape) > 0.5:
            raise ValueError("out shape is illegal")


    def forward(self, input, output_size=None):
        res = super(ConvTransposed2DWrapper, self).forward(input=input, output_size=None)
        pad_size = Padding2D.pad_size(output_size, kernel_size=self.kernel_size_, stride=self.stride_)

        self._check_shape((input.size()[2], input.size()[3]), out_shape=output_size)

        _, _, h, w = res.size()

        h_begin = pad_size[0]
        h_end = h if pad_size[1] == 0 else -pad_size[1]
        w_begin = pad_size[2]
        w_end = w if pad_size[3] == 0 else -pad_size[3]

        res = res[:, :, h_begin:h_end, w_begin:w_end]
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

    @staticmethod
    def pad_size(input_size, kernel_size, stride):
        if isinstance(kernel_size, Number):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, Number):
            stride = (stride, stride)

        stride_height, stride_width = stride
        kernel_height, kernel_width = kernel_size
        height, width = input_size

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
        return pad


if __name__ == '__main__':
    from torch.autograd import Variable

    x = Variable(torch.randn(1, 1, 3, 3))

    deconv1 = ConvTransposed2DWrapper(in_channels=1, out_channels=1, kernel_size=3, stride=2)
    print(deconv1(x, (7, 6)).size())
    # conv1 = Conv2DWrapper(in_channels=1, out_channels=1, kernel_size=3, stride=2)
    # print(conv1(x).size())
