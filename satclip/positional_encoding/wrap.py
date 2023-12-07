import torch
from torch import nn
import math

"""
Wrap encoding, as used by MacAodha et al
"""

class Wrap(nn.Module):
    def __init__(self):
        super(Wrap, self).__init__()

        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 4

    def forward(self, coords):
        # place lon lat coordinates in a -pi, pi range
        coords = torch.deg2rad(coords)

        cos_lon = torch.cos(coords[:, 0]).unsqueeze(-1)
        sin_lon = torch.sin(coords[:, 0]).unsqueeze(-1)
        cos_lat = torch.cos(coords[:, 1]).unsqueeze(-1)
        sin_lat = torch.sin(coords[:, 1]).unsqueeze(-1)

        return torch.cat((cos_lon, sin_lon, cos_lat, sin_lat), 1)