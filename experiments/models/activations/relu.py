"""ReLU activation - baseline activation function."""

import torch
import torch.nn as nn


class ReLUActivation(nn.Module):
    """ReLU activation wrapper for consistent interface.

    This is the baseline activation function. From experiments (NB16-17),
    SH + ReLU beats SIREN by +2.93%.
    """

    def __init__(self, **kwargs):
        """Initialize ReLU activation.

        Args:
            **kwargs: Ignored (for interface consistency)
        """
        super().__init__()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return self.activation(x)

    def extra_repr(self) -> str:
        return "activation=ReLU"
