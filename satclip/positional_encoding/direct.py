import torch
from torch import nn
import numpy as np
import math

from .common import _cal_freq_list

"""
Direct encoding
"""
class Direct(nn.Module):
    def __init__(self):
        super(Direct, self).__init__()

        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 2

    def forward(self, coords):
        # place lon lat coordinates in a -pi, pi range
        coords = torch.deg2rad(coords) - torch.pi
        return coords
