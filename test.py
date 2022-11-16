# Main File

import collections
from functools import reduce
from itertools import repeat
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from operator import __add__

from torch import allclose, Tensor


def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``
    """

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0,
    so we need to implement the internal torch logic manually.

    Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_,
                           range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)


# def load_model(model_name):
#     model = CNN_binary_padding().to(torch.device('cpu'))  # (device)
#     model.load_state_dict(model_name)
#     model.eval()
#     model = model.to('cpu')
#     return model
#
#
# model = load_model(model_sex_ecg)


# torch.onnx.export(model,
#                   dummy_input,
#                   "pytorch_ecg_sex.onnx",
#                   verbose=False,
#                   input_names=['ECG_matrix'],
#                   output_names=['Sex_probability'],
#                   export_params=True
#                   # opset_version=10
#                   )

# CNN architecture

class Conv2dWithSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None):
        zero_pad_2d = reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]])
        super().__init__(in_channels, out_channels, kernel_size, padding=zero_pad_2d[0], stride=stride,
                         dilation=dilation, groups=groups,
                         bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)


# model architecture

class CNN_binary_padding(nn.Module):
    def __init__(self, ):
        super(CNN_binary_padding, self).__init__()

        self.conv1 = Conv2dWithSamePadding(in_channels=1, out_channels=16, kernel_size=(9, 9))
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(16)
        self.norm3 = nn.BatchNorm2d(16)
        self.norm4 = nn.BatchNorm2d(16)
        self.norm32 = nn.BatchNorm2d(32)
        self.norm64 = nn.BatchNorm2d(64)
        self.norm1D = nn.BatchNorm1d(64)
        self.norm1D32 = nn.BatchNorm1d(32)
        self.norm128 = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((4, 1))
        self.maxpool2 = nn.MaxPool2d((2, 1))
        self.conv2 = Conv2dWithSamePadding(in_channels=16, out_channels=16, kernel_size=(7, 7))
        self.conv3 = Conv2dWithSamePadding(in_channels=16, out_channels=16, kernel_size=(5, 5))
        self.conv4 = Conv2dWithSamePadding(in_channels=16, out_channels=16, kernel_size=(5, 5))
        self.conv5 = Conv2dWithSamePadding(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv6 = Conv2dWithSamePadding(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, stride=(1, 12),
                               kernel_size=(1, 12))  # , padding='valid'
        self.fc1 = nn.Linear(in_features=1152, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)
        self.drop = nn.Dropout(0.5)
        self.flaten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.norm2(x)  # norm2
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.norm3(x)  # norm 3
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.norm4(x)  # norm 4
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.norm32(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv6(x)
        x = self.norm64(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv7(x)
        x = self.norm128(x)
        x = self.act1(x)
        # print(x.size())
        x = self.flaten(x)
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        # x = self.norm1D(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.norm1D32(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


class CNN_binary(nn.Module):
    def __init__(self, ):
        super(CNN_binary, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(9, 9), padding='same')
        self.norm1 = nn.BatchNorm2d(16)
        self.norm2 = nn.BatchNorm2d(16)
        self.norm3 = nn.BatchNorm2d(16)
        self.norm4 = nn.BatchNorm2d(16)
        self.norm32 = nn.BatchNorm2d(32)
        self.norm64 = nn.BatchNorm2d(64)
        self.norm1D = nn.BatchNorm1d(64)
        self.norm1D32 = nn.BatchNorm1d(32)
        self.norm128 = nn.BatchNorm2d(128)
        self.act1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d((4, 1))
        self.maxpool2 = nn.MaxPool2d((2, 1))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 7), padding='same')
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding='same')
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding='same')
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding='same')
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=128, stride=(1, 12), kernel_size=(1, 12), padding='valid')
        self.fc1 = nn.Linear(in_features=1152, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)
        self.drop = nn.Dropout(0.5)
        self.flaten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        print("start")
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.norm2(x)  # norm2
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.norm3(x)  # norm 3
        x = self.act1(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.norm4(x)  # norm 4
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.norm32(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv6(x)
        x = self.norm64(x)
        x = self.act1(x)
        x = self.maxpool2(x)
        x = self.conv7(x)
        x = self.norm128(x)
        x = self.act1(x)
        # print(x.size())
        x = self.flaten(x)
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        # x = self.norm1D(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.norm1D32(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


model_a = CNN_binary()
dummy_input = torch.randn(1, 1, 5000, 12)
model_a.eval()
out_a = model_a(dummy_input)
PATH = "model.pth"
torch.save(model_a.state_dict(), PATH)
model_b = CNN_binary_padding()
model_b.load_state_dict(torch.load(PATH))
model_b.eval()
out_b = model_b(dummy_input)
torch.allclose(out_a, out_b)
torch.onnx.export(
    model=model_b,
    args=dummy_input,
    f='ecg.onnx',
)

# def calc_same_padding(kernel_size, stride, input_size):
#     if isinstance(kernel_size, Sequence):
#         kernel_size = kernel_size[0]
#
#     if isinstance(stride, Sequence):
#         stride = stride[0]
#
#     if isinstance(input_size, Sequence):
#         input_size = input_size[0]
#
#     pad = ((stride - 1) * input_size - stride + kernel_size) / 2
#     return int(pad)
#
#
# def replace_conv2d_with_same_padding(m: nn.Module, input_size=512):
#     if isinstance(m, nn.Conv2d):
#         if m.padding == "same":
#             m.padding = calc_same_padding(
#                 kernel_size=m.kernel_size,
#                 stride=m.stride,
#                 input_size=input_size
#             )
#
#
# model = CNN_binary()
# dummy_input = torch.randn(1, 1, 5000, 12)
# model(dummy_input)
# torch.onnx.export(
#     model=model,
#     args=dummy_input,
#     f='ecg.onnx',
# )
# # input_names=['input'],
# # # opset_version=20,  # same error with any other version <= 14
# # output_names=['output'])
#
# model.apply(lambda m: replace_conv2d_with_same_padding(m, 512))
