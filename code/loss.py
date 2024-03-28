import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, labels):
        inputs = inputs.view(-1)
        labels = labels.view(-1)
        dice_coefficient = self.dice(inputs, labels)
        return 1 - dice_coefficient

    def dice(self, inputs, labels, epsilon=1e-6):
        intersect = inputs*labels
        union = inputs+labels
        return 2.0 * (inputs * labels).sum() / ((inputs + labels).sum() + epsilon)