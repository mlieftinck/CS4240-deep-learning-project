import torch
from train import train, test
from model import nsn, ndn
import torch.nn as nn
from load_data import preprocessing
from view_images import view_image
from loss import DiceLoss
import time
import os


def run_training():
    images, labels = preprocessing("NSN")
    print(f'------------ RUNNING ------------')
    images = images.float()
    labels = labels.long()

    net = nsn()

    optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = DiceLoss()
    time_start = time.time()

    train_loss, train_acc = train(images, labels, net, optimizer, criterion)
    test_loss, test_acc, output_array = test(images, labels, net, criterion)
    time_end = time.time() - time_start

    print(output_array.size())
    print(f'test loss: {test_loss}, test accuracy = {test_acc}')
    print(f"Total elapsed time: {time_end // 60:.0f} min {time_end % 60:.0f} sec")

    # Save trained model
    parent_directory = os.path.dirname(os.getcwd())
    os.makedirs(parent_directory + '/trained_models/', exist_ok=True)
    torch.save(net.state_dict(), parent_directory + '/trained_models/nsn.pth')

    view_image(output_array.cpu().detach().numpy()[0])
    # view_image(labels[0].cpu().detach().numpy()[0])

    return output_array.cpu().detach().numpy()


def load_trained_model():
    model = nsn()
    parent_dir = os.path.dirname(os.getcwd())
    model.load_state_dict(torch.load(parent_dir + '/trained_models/trained_nsn.pth'))


if __name__ == "__main__":
    o = run_training()
    view_image(o[0])
