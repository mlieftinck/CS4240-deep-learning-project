import torch
import torch.nn as nn
from train import train


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


if __name__ == "__main__":
    image = torch.zeros((1, 1, 1, 112, 114, 112))
    labels = torch.zeros((1, 1, 112, 114, 112))
    labels = labels.long()
    net = nsn()
    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = nn.CrossEntropyLoss()
    train(image, labels, net, optimizer, criterion)

