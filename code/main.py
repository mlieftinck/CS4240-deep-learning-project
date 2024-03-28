import torch
from train import train, test
from model import nsn
import torch.nn as nn
from load_data import preprocessing
from view_images import view_image
from loss import DiceLoss


if __name__ == "__main__":
    images, labels = preprocessing()
    print(f'------------ RUNNING ------------')
    # Try for only a singe timeseries
    image = images[0]

    image = torch.unsqueeze(image, 0)
    image = image.float()
    labels = labels.long()
    net = nsn()
    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = nn.CrossEntropyLoss()
    # criterion = DiceLoss()
    train_loss, train_acc = train(image, labels, net, optimizer, criterion)
    test_loss, test_acc, output_array = test(image, labels, net, criterion)
    print(output_array.size())
    print(f'test loss: {test_loss}, test accuracy = {test_acc}')
    view_image(output_array.cpu().detach().numpy()[0])
