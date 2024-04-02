import torch
from train import train, test
from model import nsn, ndn
import torch.nn as nn
from load_data import preprocessing
from view_images import view_image
from loss import DiceLoss
import os


def run_training():
    images, labels = preprocessing("NSN")
    print(f'------------ RUNNING ------------')
    images = images.float()
    labels = labels.long()

    net = nsn()

    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = DiceLoss()

    train_loss, train_acc = train(images, labels, net, optimizer, criterion)
    test_loss, test_acc, output_array = test(images, labels, net, criterion)

    print(output_array.size())
    print(f'test loss: {test_loss}, test accuracy = {test_acc}')

    os.makedirs('output/', exist_ok=True)
    torch.save(net.state_dict(), 'output/trained_nsn.pth')
    return output_array.cpu().detach().numpy()


def load_trained_model():
    model = nsn()
    model.load_state_dict(torch.load('output/trained_nsn.pth'))


if __name__ == "__main__":
    o = run_training()
    view_image(o[0])
