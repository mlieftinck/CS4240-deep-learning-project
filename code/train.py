import torch
import numpy


def train(data, labels, net, optimizer, criterion, device):
    avg_loss = 0
    correct = 0
    total = 0
    data_len = len(data)
    for i in range(data_len):
        # get batch
        inp = data[i]
        truth = labels[i]

        inp = torch.unsqueeze(inp, 0)
        truth = torch.unsqueeze(truth, 0)

        # to GPU
        inp = inp.to(device)
        truth = truth.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(inp)
        loss = criterion(output, truth)
        loss.backward()
        optimizer.step()

        # keep track of loss and accuracy
        avg_loss += loss
        _, predicted = torch.max(output.data, 1)
        total += truth.flatten().size(0)
        correct += (predicted == truth).sum().item()

        del inp, truth, predicted

    return avg_loss / len(data), 100 * correct / total


def test(test_data, test_labels, net, criterion):
    avg_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(test_data)):
            # get batch
            inp = test_data[i]
            lab = test_labels[i]

            inp = torch.unsqueeze(inp, 0)
            lab = torch.unsqueeze(lab, 0)

            # only forward pass
            outputs = net(inp)
            loss = criterion(outputs, lab)

            # keep track of loss and accuracy
            avg_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += lab.flatten().size(0)
            correct += (predicted == lab).sum().item()

        return avg_loss / len(test_data), 100 * correct / total, predicted


def dice_loss(output, ground_truth):
    """
    dice loss function as an objective function can suppress the influence of dataset label imbalance.
    :param output: output value from the final layer
    :param ground_truth:
    :return: dice loss
    """
    return 2 * torch.sum(output*ground_truth) / (torch.sum(output**2) + torch.sum(ground_truth**2))