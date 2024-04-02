import torch
import torch.nn as nn
import torch.nn.functional as F


class nsn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cov1 = nn.Conv3d(1, 16, 3, 1, 1)
        self.batchnorm1 = nn.BatchNorm3d(16)
        self.relu1 = nn.ReLU()

        self.cov2 = nn.Conv3d(16, 32, 3, 1, 1)
        self.batchnorm2 = nn.BatchNorm3d(32)
        self.relu2 = nn.ReLU()

        self.max_pool1 = nn.MaxPool3d(2, 2)
        self.cov3 = nn.Conv3d(32, 32, 3, 1, 1)
        self.batchnorm3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU()

        self.cov4 = nn.Conv3d(32, 64, 3, 1, 1)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.relu4 = nn.ReLU()

        self.max_pool2 = nn.MaxPool3d(2, 2)

        self.cov5 = nn.Conv3d(64, 64, 3, 1, 1)
        self.batchnorm5 = nn.BatchNorm3d(32)
        self.relu5 = nn.ReLU()

        self.cov6 = nn.Conv3d(64, 128, 3, 1, 1)
        self.batchnorm6 = nn.BatchNorm3d(128)
        self.relu6 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose3d(128, 128, 2, 2, 0)
        self.relu7 = nn.ReLU()

        self.last_conv = nn.Conv3d(64, 2, 1, 1, 0)


    def forward(self, x):
        h = self.relu1(self.batchnorm1(self.cov1(x)))
        h = self.relu2(self.batchnorm2(self.cov2(h)))
        # h = self.max_pool1(h)
        h = self.relu3(self.batchnorm3(self.cov3(h)))
        h = self.relu4(self.batchnorm4(self.cov4(h)))
        # h = self.max_pool1(h)
        # h = self.relu5(self.batchnorm5(self.cov3(h)))
        # h = self.relu6(self.batchnorm6(self.cov4(h)))
        # h = self.relu7(self.deconv1(h)))

        h = self.last_conv(h)
        return F.softmax(h, 1)


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
