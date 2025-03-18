import torch
from torch import nn
from satclip.positional_encoding.spherical_harmonics_ylm import SH as SH_analytic
from satclip.positional_encoding.spherical_harmonics_closed_form import SH as SH_closed_form

"""
Spherical Harmonics locaiton encoder
"""
class SphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys: int = 10, harmonics_calculation="analytic"):
        """
        legendre_polys: determines the number of legendre polynomials.
                        more polynomials lead more fine-grained resolutions
        calculation of spherical harmonics:
            analytic uses pre-computed equations. This is exact, but works only up to degree 50,
            closed-form uses one equation but is computationally slower (especially for high degrees)
        """
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M

        if harmonics_calculation == "closed-form":
            self.SH = SH_closed_form
        elif harmonics_calculation == "analytic":
            self.SH = SH_analytic

    def forward(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = self.SH(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                Y.append(y)

        return torch.stack(Y,dim=-1)
