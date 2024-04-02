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
        # CBR, CBR, Pooling
        self.conv1 = nn.Conv3d(input_channels, 12, 5, 1, 5 // 2)
        self.bnorm1 = nn.BatchNorm3d(12)
        self.conv2 = nn.Conv3d(12, 24, 5, 1, 5 // 2)
        self.bnorm2 = nn.BatchNorm3d(24)
        # CBR, CBR, Pooling
        self.conv4 = nn.Conv3d(24, 24, 5, 1, 5 // 2)
        self.bnorm4 = nn.BatchNorm3d(24)
        self.conv5 = nn.Conv3d(24, 48, 5, 1, 5 // 2)
        self.bnorm5 = nn.BatchNorm3d(48)
        # CBR, CBR, Pooling
        self.conv7 = nn.Conv3d(48, 48, 5, 1, 5 // 2)
        self.bnorm7 = nn.BatchNorm3d(48)
        self.conv8 = nn.Conv3d(48, 96, 5, 1, 5 // 2)
        self.bnorm8 = nn.BatchNorm3d(96)
        # CBR, CBR, Pooling
        self.conv10 = nn.Conv3d(96, 96, 5, 1, 5 // 2)
        self.bnorm10 = nn.BatchNorm3d(96)
        self.conv11 = nn.Conv3d(96, 192, 5, 1, 5 // 2)
        self.bnorm11 = nn.BatchNorm3d(192)
        # CBR, CBR Deconvolution, Concatenation
        self.conv13 = nn.Conv3d(192, 192, 5, 1, 5 // 2)
        self.bnorm13 = nn.BatchNorm3d(192)
        self.conv14 = nn.Conv3d(192, 384, 5, 1, 5 // 2)
        self.bnorm14 = nn.BatchNorm3d(384)
        self.dconv15 = nn.ConvTranspose3d(384, 384, 2, 2, 0)
        # CBR, CBR Deconvolution, Concatenation
        self.conv17 = nn.Conv3d(384, 192, 5, 1, 5 // 2)
        self.bnorm17 = nn.BatchNorm3d(192)
        self.conv18 = nn.Conv3d(192, 192, 5, 1, 5 // 2)
        self.bnorm18 = nn.BatchNorm3d(192)
        self.dconv19 = nn.ConvTranspose3d(192, 192, 2, 2, 0)
        # CBR, CBR Deconvolution, Concatenation
        self.conv21 = nn.Conv3d(192, 96, 5, 1, 5 // 2)
        self.bnorm21 = nn.BatchNorm3d(96)
        self.conv22 = nn.Conv3d(96, 96, 5, 1, 5 // 2)
        self.bnorm22 = nn.BatchNorm3d(96)
        self.dconv23 = nn.ConvTranspose3d(96, 96, 2, 2, 0)
        # CBR, CBR Deconvolution, Concatenation
        self.conv25 = nn.Conv3d(96, 48, 5, 1, 5 // 2)
        self.bnorm25 = nn.BatchNorm3d(48)
        self.conv26 = nn.Conv3d(48, 48, 5, 1, 5 // 2)
        self.bnorm26 = nn.BatchNorm3d(48)
        self.dconv27 = nn.ConvTranspose3d(48, 48, 2, 2, 0)
        # CBR, CBR, Convolution
        self.conv29 = nn.Conv3d(48, 24, 5, 1, 5 // 2)
        self.bnorm29 = nn.BatchNorm3d(24)
        self.conv30 = nn.Conv3d(24, 24, 5, 1, 5 // 2)
        self.bnorm30 = nn.BatchNorm3d(24)
        self.conv31 = nn.Conv3d(24, 2, 1, 1, 0)

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
        cbr11 = self.relu(self.bnorm11(self.conv11(cbr10)))
        mp12 = self.pool(cbr11)
        del cbr10, cbr11, mp9
        cbr13 = self.relu(self.bnorm13(self.conv13(mp12)))
        cbr14 = self.relu(self.bnorm14(self.conv14(cbr13)))
        dconv15 = self.dconv15(cbr14)
        del cbr13, cbr14, mp12
        conc16 = dconv15
        cbr17 = self.relu(self.bnorm17(self.conv17(conc16)))
        cbr18 = self.relu(self.bnorm18(self.conv18(cbr17)))
        dconv19 = self.dconv19(cbr18)
        del cbr17, cbr18, dconv15, conc16
        conc20 = dconv19
        cbr21 = self.relu(self.bnorm21(self.conv21(conc20)))
        cbr22 = self.relu(self.bnorm22(self.conv22(cbr21)))
        dconv23 = self.dconv23(cbr22)
        del cbr21, cbr22, dconv19, conc20
        conc24 = dconv23
        cbr25 = self.relu(self.bnorm25(self.conv25(conc24)))
        cbr26 = self.relu(self.bnorm26(self.conv26(cbr25)))
        dconv27 = self.dconv27(cbr26)
        del cbr25, cbr26, dconv23, conc24
        conc28 = dconv27
        cbr29 = self.relu(self.bnorm29(self.conv29(conc28)))
        cbr30 = self.relu(self.bnorm30(self.conv30(cbr29)))
        pred = self.conv31(cbr30)

        return F.softmax(pred, 1)
