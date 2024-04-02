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

        # Single-use layers
        self.conv1 = nn.Conv3d(input_channels, 12, 5, 1, 5 // 2)
        self.bnorm1 = nn.BatchNorm3d(12)

        self.conv2 = nn.Conv3d(12, 24, 5, 1, 5 // 2)
        self.bnorm2 = nn.BatchNorm3d(24)

        self.conv4 = nn.Conv3d(24, 24, 5, 1, 5 // 2)
        self.bnorm4 = nn.BatchNorm3d(24)

        self.conv5 = nn.Conv3d(24, 48, 5, 1, 5 // 2)
        self.bnorm5 = nn.BatchNorm3d(48)

        self.conv7 = nn.Conv3d(48, 48, 5, 1, 5 // 2)
        self.bnorm7 = nn.BatchNorm3d(48)

        self.conv8 = nn.Conv3d(48, 96, 5, 1, 5 // 2)
        self.bnorm8 = nn.BatchNorm3d(96)

        self.conv10 = nn.Conv3d(96, 96, 5, 1, 5 // 2)
        self.bnorm10 = nn.BatchNorm3d(96)

        self.conv11 = nn.Conv3d(96, 192, 5, 1, 5 // 2)
        self.bnorm11 = nn.BatchNorm3d(192)

        self.conv13 = nn.Conv3d(192, 192, 5, 1, 5 // 2)
        self.bnorm13 = nn.BatchNorm3d(192)

        self.conv14 = nn.Conv3d(192, 384, 5, 1, 5 // 2)
        self.bnorm14 = nn.BatchNorm3d(384)

        self.dconv15 = nn.ConvTranspose3d(384, 384, 2, 2, 0)

        current_final_channel_number = 48
        self.conv31 = nn.Conv3d(current_final_channel_number, 2, 1, 1, 0)

        # Multi-use layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(2, 2)

    def forward(self, x):
        cbr1 = self.relu(self.bnorm1(self.conv1(x)))
        cbr2 = self.relu(self.bnorm2(self.conv2(cbr1)))
        mp3 = self.pool(cbr2)
        del cbr1, cbr2
        cbr4 = self.relu(self.bnorm4(self.conv4(mp3)))
        cbr5 = self.relu(self.bnorm5(self.conv5(cbr4)))
        mp6 = self.pool(cbr5)
        del cbr4, cbr5, mp3
        cbr7 = self.relu(self.bnorm7(self.conv7(mp6)))
        cbr8 = self.relu(self.bnorm8(self.conv8(cbr7)))
        mp9 = self.pool(cbr8)
        del cbr7, cbr8, mp6
        cbr10 = self.relu(self.bnorm10(self.conv10(mp9)))
        cbr11 = self.relu(self.bnorm10(self.conv11(cbr10)))
        mp12 = self.pool(cbr11)
        del cbr10, cbr11, mp9
        cbr13 = self.relu(self.bnorm13(self.conv13(mp12)))
        cbr14 = self.relu(self.bnorm14(self.conv14(cbr13)))
        dconv15 = self.dconv15(cbr14)
        del cbr13, cbr14, mp12

        pred = self.conv31(dconv15)
        # test was dconv(cbr5) -> alles zwart...
        return F.softmax(pred, 1)
