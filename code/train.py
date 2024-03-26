import torch


def train(data, labels, net, optimizer, criterion):
    for i in range(len(data)):
        inp = data[i]
        truth = labels[i]
        # zero the parameter gradients
        optimizer.zero_grad()
        output = net(inp)
        loss = criterion(output, truth)
        loss.backward()
        optimizer.step()


def dice_loss(output, ground_truth):
    """
    dice loss function as an objective function can suppress the influence of dataset label imbalance.
    :param output: output value from the final layer
    :param ground_truth:
    :return: dice loss
    """
    return 2 * torch.sum(output*ground_truth) / (torch.sum(output**2) + torch.sum(ground_truth**2))