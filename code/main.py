import torch
from train import train, test
from model import nsn, ndn
import torch.nn as nn
from load_data import preprocessing
from view_images import view_image
from loss import DiceLoss


if __name__ == "__main__":
    images, labels = preprocessing("NDN")
    print(f'------------ RUNNING ------------')

    # images = torch.unsqueeze(images, 0)
    images = images.float()
    labels = labels.long()
    net = ndn()
    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = nn.CrossEntropyLoss()
    # criterion = DiceLoss()
    train_loss, train_acc = train(images, labels, net, optimizer, criterion)
    test_loss, test_acc, output_array = test(images, labels, net, criterion)
    print(output_array.size())
    print(f'test loss: {test_loss}, test accuracy = {test_acc}')
    view_image(output_array.cpu().detach().numpy()[0])
    # view_image(labels[0].cpu().detach().numpy()[0])
