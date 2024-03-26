import torch
from train import train
from model import nsn
import torch.nn as nn


if __name__ == "__main__":
    image = torch.zeros((1, 1, 1, 112, 114, 112))
    labels = torch.zeros((1, 1, 112, 114, 112))
    labels = labels.long()
    net = nsn()
    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = nn.CrossEntropyLoss()
    train(image, labels, net, optimizer, criterion)