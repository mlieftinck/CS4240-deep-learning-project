import torch
import torch.nn as nn
import torch.nn.functional as F


class nsn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cbr1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.batchnorm1 = nn.BatchNorm3d(16)
        # self.max_pool1 =
        # self.deconv1 =
        # self. concat1 =

    def forward(self, x):
        h = self.batchnorm1(self.cbr1(x))
        return h


class ndn(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        self.conv1 = nn.Conv3d(input_channels, 12, 5, 1, 5 // 2)
        self.bnorm1 = nn.BatchNorm3d(12)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(12, 24, 5, 1, 5 // 2)
        self.bnorm2 = nn.BatchNorm3d(24)
        self.relu2 = nn.ReLU()

        self.conv31 = nn.Conv3d(24, 2, 1, 1, 0)

    def forward(self, x):
        cbr1 = self.relu1(self.bnorm1(self.conv1(x)))
        cbr2 = self.relu2(self.bnorm2(self.conv2(cbr1)))

        pred = self.conv31(cbr2)
        return F.softmax(pred, 1)
