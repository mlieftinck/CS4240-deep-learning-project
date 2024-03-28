import torch
from train import train, test
from model import nsn
import torch.nn as nn
from load_data import preprocessing
from view_images import view_image


if __name__ == "__main__":
    image, labels = preprocessing()
    print(f'------------ RUNNING ------------')
    image = torch.unsqueeze(image, 0)
    image = image.float()
    labels = labels.long()
    net = nsn()
    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = train(image, labels, net, optimizer, criterion)
    test_loss, test_acc, output_array = test(image, labels, net, criterion)
    print(output_array.size())
    print(f'test loss: {test_loss}, test accuracy = {test_acc}')
    view_image(output_array.cpu().detach().numpy()[0])
