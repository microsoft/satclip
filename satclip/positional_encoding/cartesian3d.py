import torch
from torch import nn
import math

"""
3D Cartesian
"""
class Cartesian3D(nn.Module):
    def __init__(self):
        super(Cartesian3D, self).__init__()

        # adding this class variable is important to determine
        # the dimension of the follow-up neural network
        self.embedding_dim = 3

    def forward(self, coords):
        # place lon lat coordinates in a -pi, pi range
        coords = torch.deg2rad(coords)

        cos_lon = torch.cos(coords[:, 0]).unsqueeze(-1)
        sin_lon = torch.sin(coords[:, 0]).unsqueeze(-1)
        cos_lat = torch.cos(coords[:, 1]).unsqueeze(-1)
        sin_lat = torch.sin(coords[:, 1]).unsqueeze(-1)

        return torch.cat((cos_lon * cos_lat, sin_lon * cos_lat, sin_lat), 1)
