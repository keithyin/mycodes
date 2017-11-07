from torch import nn
import torch
from torch.nn import functional as F
import math


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
    pad1 = Padding2D(stride=2, kernel=3)
    t4d = torch.Tensor(3, 3, 224, 224)
    print(pad1(t4d).size())


if __name__ == '__main__':
    main()
