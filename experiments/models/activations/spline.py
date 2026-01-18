"""Learnable Spline Activation Function.

Based on experiments in NB18, optimal configuration:
- k=15 knots
- ReLU initialization
- Input range (-3, 3)
- Fixed knot positions

Key findings:
- Spline beats SIREN by +1.88%
- ReLU still wins by +0.63% over best spline
- Zero initialization is catastrophic
- Diminishing returns for k>20
"""

import torch
import torch.nn as nn
from typing import Literal, Tuple


class SplineActivation(nn.Module):
    """Learnable piecewise-linear spline activation function.

    The spline is defined by knot points with learnable y-values.
    Interpolation between knots is linear.

    Optimal configuration (from NB18):
    - n_knots=15
    - init='relu'
    - input_range=(-3.0, 3.0)
    - learnable_positions=False
    """

    def __init__(
        self,
        n_knots: int = 15,
        input_range: Tuple[float, float] = (-3.0, 3.0),
        init: Literal["relu", "linear", "zero", "tanh", "gelu"] = "relu",
        learnable_positions: bool = False,
        **kwargs,
    ):
        """Initialize spline activation.

        Args:
            n_knots: Number of knot points (5-20 recommended, 15 optimal)
            input_range: (min, max) range for knot positions
            init: Initialization strategy:
                - 'relu': Initialize to approximate ReLU (recommended)
                - 'linear': Initialize to identity function
                - 'zero': Initialize to zero (NOT recommended, catastrophic)
                - 'tanh': Initialize to approximate tanh
                - 'gelu': Initialize to approximate GELU
            learnable_positions: Whether knot x-positions are learnable
            **kwargs: Ignored
        """
        super().__init__()
        self.n_knots = n_knots
        self.input_range = input_range
        self.init_type = init
        self.learnable_positions = learnable_positions

        # Initialize knot x-positions
        knot_x = torch.linspace(input_range[0], input_range[1], n_knots)

        # Initialize knot y-values based on init strategy
        if init == "relu":
            knot_y = torch.relu(knot_x)
        elif init == "linear":
            knot_y = knot_x.clone()
        elif init == "zero":
            knot_y = torch.zeros(n_knots)
        elif init == "tanh":
            knot_y = torch.tanh(knot_x)
        elif init == "gelu":
            knot_y = torch.nn.functional.gelu(knot_x)
        else:
            raise ValueError(f"Unknown init: {init}")

        if learnable_positions:
            self.knot_x = nn.Parameter(knot_x)
        else:
            self.register_buffer("knot_x", knot_x)

        self.knot_y = nn.Parameter(knot_y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spline activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Activated tensor of same shape
        """
        # Clamp to valid range
        x_clamped = torch.clamp(x, self.input_range[0], self.input_range[1])

        # Normalize to [0, n_knots-1] range
        x_norm = (x_clamped - self.knot_x[0]) / (self.knot_x[-1] - self.knot_x[0])
        x_idx = x_norm * (self.n_knots - 1)

        # Get surrounding knot indices
        idx_low = torch.floor(x_idx).long()
        idx_high = torch.clamp(idx_low + 1, max=self.n_knots - 1)
        idx_low = torch.clamp(idx_low, min=0, max=self.n_knots - 1)

        # Linear interpolation weight
        weight = x_idx - idx_low.float()

        # Interpolate between knots
        y_low = self.knot_y[idx_low]
        y_high = self.knot_y[idx_high]

        return y_low + weight * (y_high - y_low)

    def extra_repr(self) -> str:
        return (
            f"n_knots={self.n_knots}, "
            f"input_range={self.input_range}, "
            f"init={self.init_type}, "
            f"learnable_positions={self.learnable_positions}"
        )

    def get_knot_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current knot positions and values for visualization.

        Returns:
            Tuple of (knot_x, knot_y) tensors
        """
        return self.knot_x.detach().cpu(), self.knot_y.detach().cpu()


class SplineLayer(nn.Module):
    """Linear layer followed by spline activation.

    Convenience module combining nn.Linear with SplineActivation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_knots: int = 15,
        input_range: Tuple[float, float] = (-3.0, 3.0),
        init: str = "relu",
        use_bias: bool = True,
        **kwargs,
    ):
        """Initialize spline layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            n_knots: Number of spline knots
            input_range: Input range for spline
            init: Spline initialization strategy
            use_bias: Whether to use bias in linear layer
            **kwargs: Additional args for SplineActivation
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.activation = SplineActivation(
            n_knots=n_knots,
            input_range=input_range,
            init=init,
            **kwargs,
        )

        # Initialize linear weights
        nn.init.kaiming_normal_(self.linear.weight)
        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation and spline activation.

        Args:
            x: Input tensor

        Returns:
            Activated output
        """
        return self.activation(self.linear(x))
