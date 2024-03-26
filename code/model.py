import torch
import torch.nn as nn


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


