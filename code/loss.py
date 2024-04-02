import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, labels):
        input_object_channel = inputs[:, 1, :, :, :]
        inputs = input_object_channel.reshape(-1)
        labels = labels.reshape(-1)
        dice_coefficient = self.dice(inputs, labels)
        return 1 - dice_coefficient

    def dice(self, inputs, labels, epsilon=1e-6):
        # intersect = inputs * labels
        # union = inputs + labels
        return 2.0 * (inputs * labels).sum() / ((inputs + labels).sum() + epsilon)
