import torch
from train import train, test
from model import nsn, ndn, mini_ndn
import torch.nn as nn
from load_data import preprocessing
from view_images import view_image
from loss import DiceLoss
import time
import os


def run_training(project_dir, model="NSN", save=False):
    NDN = False
    if model == "NDN":
        NDN = True
    images, labels = preprocessing(model, project_dir)
    print(f'------------ RUNNING ------------')
    images = images.float()
    labels = labels.long()
    if NDN:
        net = ndn()
    else:
        net = nsn()

    if NDN:
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-1)
    else:
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
    if save:
        parent_directory = os.path.dirname(os.getcwd())
        os.makedirs(parent_directory + '/trained_models/', exist_ok=True)
        torch.save(net.state_dict(), parent_directory + '/trained_models/' + model + '.pth')

    view_image(output_array.cpu().detach().numpy()[0])
    # view_image(labels[0].cpu().detach().numpy()[0])

    return output_array.cpu().detach().numpy()


def load_trained_model(model, model_name):
    parent_dir = os.path.dirname(os.getcwd())
    model.load_state_dict(torch.load(parent_dir + '/trained_models/' + model_name))


def load_processed_data(path):
    project_dir = os.path.dirname(os.getcwd())
    return torch.load(os.path.join(project_dir, path), map_location=torch.device("cpu"))


def run_training_mini_ndn():
    batches = 3
    timeseries_full = load_processed_data("preprocessed_data/input/train.pt")
    labels_full = load_processed_data("preprocessed_data/labels/train/NDN.pt")
    timeseries = timeseries_full[:batches]
    labels = labels_full[:batches]
    del timeseries_full, labels_full
    print(f'------------ RUNNING ------------')
    timeseries = timeseries.float()
    labels = labels.long()
    net = mini_ndn()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-1)
    criterion = DiceLoss()

    time_start = time.time()
    train_loss, train_acc = train(timeseries, labels, net, optimizer, criterion)
    test_loss, test_acc, output_array = test(timeseries, labels, net, criterion)
    time_end = time.time() - time_start

    print(output_array.size())
    print(f'test loss: {test_loss}, test accuracy = {test_acc}')
    print(f"Total elapsed time: {time_end // 60:.0f} min {time_end % 60:.0f} sec")

    view_image(output_array.cpu().detach().numpy()[0])
    # view_image(labels[0].cpu().detach().numpy()[0])

    return output_array.cpu().detach().numpy()


if __name__ == "__main__":
    # o = run_training(os.getcwd())
    o = run_training_mini_ndn()
    view_image(o[0])
