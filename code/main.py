import torch
from train import train
from model import nsn
import torch.nn as nn
from load_data import preprocessing


if __name__ == "__main__":
    image, labels = preprocessing()
    image = torch.unsqueeze(image, 0)
    # image = torch.zeros((1, 1, 1, 112, 114, 112))
    # labels = torch.zeros((1, 1, 112, 114, 112))
    image = image.float()
    labels = labels.long()
    net = nsn()
    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = nn.CrossEntropyLoss()
    l = train(image, labels, net, optimizer, criterion)
    print(l)
