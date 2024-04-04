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

        self.conv7 = nn.Conv3d(192, 64, 3, 1, 1)
        self.batchnorm7 = nn.BatchNorm3d(64)
        self.relu8 = nn.ReLU()

        self.conv8 = nn.Conv3d(64, 64, 3, 1, 1)
        self.batchnorm8 = nn.BatchNorm3d(64)
        self.relu9 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose3d(64, 64, 2, 2, 0)
        self.relu10 = nn.ReLU()

        self.conv9 = nn.Conv3d(64, 32, 3, 1, 1)
        self.batchnorm9 = nn.BatchNorm3d(32)
        self.relu11 = nn.ReLU()

        self.conv10 = nn.Conv3d(32, 32, 3, 1, 1)
        self.batchnorm10 = nn.BatchNorm3d(32)
        self.relu11 = nn.ReLU()

        self.last_conv = nn.Conv3d(32, 2, 1, 1, 0)

    def forward(self, x):
        h = self.relu1(self.batchnorm1(self.cov1(x)))
        out_layer_2 = self.relu2(self.batchnorm2(self.cov2(h)))
        h = self.max_pool1(out_layer_2)
        h = self.relu3(self.batchnorm3(self.cov3(h)))
        out_layer_4 = self.relu4(self.batchnorm4(self.cov4(h)))
        h = self.max_pool1(out_layer_4)
        h = self.relu5(self.batchnorm5(self.cov3(h)))
        h = self.relu6(self.batchnorm6(self.cov4(h)))
        h = self.relu7(self.deconv1(h))
        h = torch.concat([out_layer_4, h], 1)
        del out_layer_4
        h = self.relu8(self.batchnorm7(self.conv7(h)))
        h = self.relu9(self.batchnorm8(self.conv8(h)))
        h = self.relu10(self.deconv2(h))
        h = torch.concat([out_layer_2, h], 1)
        del out_layer_2
        h = self.relu11(self.batchnorm9(self.conv9(h)))
        h = self.relu12(self.batchnorm10(self.conv10(h)))
        h = self.last_conv(h)
        return F.softmax(h, 1)


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


class mini_ndn(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        # Single-use layers
        # CBR, CBR, Pooling
        self.conv1 = nn.Conv3d(input_channels, 12, 5, 1, 5 // 2)
        self.bnorm1 = nn.BatchNorm3d(12)
        self.conv2 = nn.Conv3d(12, 24, 5, 1, 5 // 2)
        self.bnorm2 = nn.BatchNorm3d(24)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.dconv4 = nn.ConvTranspose3d(384, 384, 2, 2, 0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # CBR, CBR, Pooling, Deconvolution
        cbr1 = self.relu(self.bnorm1(self.conv1(x)))
        cbr2 = self.relu(self.bnorm2(self.conv2(cbr1)))
        mp3 = self.pool3(cbr2)
        dconv4 = self.dconv4(mp3)

        return F.softmax(dconv4, 1)
