import torch
from train import train
from model import nsn, ndn
from loss import DiceLoss
import time
import os


def load_trained_model(model, model_name):
    model.load_state_dict(torch.load('C:/Users/raelh/Documents/Master CS/Deep learning/Project/Code/CS4240-deep-learning-project/data/NSN(1).pth', map_location=torch.device('cpu')))


def run_training(images, labels, model="NSN", save=False):
    print(f'------------ RUNNING ------------')
    NDN = False
    if model == "NDN":
        NDN = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if NDN:
        net = ndn().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=5e-1)
    else:
        net = nsn()
        net.load_state_dict(torch.load('/kaggle/input/nsntensor/NSN.pth'))
        net = net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=5e-1)
    criterion = DiceLoss()

    losses = []
    accuracies = []

    for i in range(40):
        time_start = time.time()
        train_loss, train_acc = train(images, labels, net, optimizer, criterion, device)

        time_end = time.time() - time_start

        losses.append(train_loss)
        accuracies.append(train_acc)

        print(f"Epoch: {i}, Total elapsed time: {time_end // 60:.0f} min {time_end % 60:.0f} sec")
        print(f'train loss: {train_loss}, train accuracy = {train_acc}')

    # Save trained model
    if save:
        parent_directory = os.path.dirname(os.getcwd())
        os.makedirs(parent_directory + '/working/trained_models/', exist_ok=True)
        torch.save(net.state_dict(), parent_directory + '/working/trained_models/' + model + '.pth')
    return torch.tensor(losses), torch.tensor(accuracies)


if __name__ == "__main__":
    images = torch.load('/kaggle/input/micedatapreprocessed/batch_of_images.pt')
    labels_nsn = torch.load('/kaggle/input/micedatapreprocessed/batch_of_truths_NSN.pt')
    loss, acc = run_training(images, labels_nsn, model="NSN", save=True)
    loss = loss.cpu().detach().numpy()
    acc = acc.cpu().detach().numpy()
    loss = torch.tensor(loss)
    acc = torch.tensor(acc)
    parent_directory = os.path.dirname(os.getcwd())
    model = 'NSN'
    os.makedirs(parent_directory + '/working/trained_models/', exist_ok=True)
    torch.save(loss, parent_directory + '/working/trained_models/' + model + 'loss.pth')
    torch.save(acc, parent_directory + '/working/trained_models/' + model + 'acc.pth')
