"""SIREN (Sinusoidal Representation Network) activation.

Based on: "Implicit Neural Representations with Periodic Activation Functions"
Sitzmann et al., NeurIPS 2020

SIREN uses sine activations with special initialization to enable learning
high-frequency functions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    """Sine activation function with learnable frequency."""

    def __init__(self, w0: float = 1.0):
        """Initialize Sine activation.

        Args:
            w0: Frequency multiplier (higher = higher frequencies learned)
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sine activation.

        Args:
            x: Input tensor

        Returns:
            sin(w0 * x)
        """
        return torch.sin(self.w0 * x)

    def extra_repr(self) -> str:
        return f"w0={self.w0}"


class SIRENActivation(nn.Module):
    """SIREN activation with proper initialization.

    This implements a single SIREN layer with:
    - Linear transformation
    - Sine activation
    - Proper SIREN initialization
    """

    def __init__(
        self,
        in_features: int = None,
        out_features: int = None,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        is_first: bool = False,
        use_bias: bool = True,
        c: float = 6.0,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Initialize SIREN activation layer.

        Args:
            in_features: Input dimension (optional, for standalone use)
            out_features: Output dimension (optional, for standalone use)
            w0: Frequency multiplier for hidden layers
            w0_initial: Frequency multiplier for first layer (typically higher)
            is_first: Whether this is the first layer in the network
            use_bias: Whether to use bias in linear layer
            c: Constant for SIREN initialization
            dropout: Dropout probability
            **kwargs: Ignored
        """
        super().__init__()
        self.w0 = w0_initial if is_first else w0
        self.is_first = is_first
        self.c = c
        self.dropout_p = dropout

        # Optional linear layer for standalone use
        self.linear = None
        if in_features is not None and out_features is not None:
            self.linear = nn.Linear(in_features, out_features, bias=use_bias)
            self._init_weights(in_features)

        self.activation = Sine(self.w0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _init_weights(self, in_features: int):
        """Initialize weights using SIREN initialization scheme.

        Args:
            in_features: Input dimension for computing std
        """
        if self.linear is None:
            return

        if self.is_first:
            w_std = 1.0 / in_features
        else:
            w_std = math.sqrt(self.c / in_features) / self.w0

        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if self.linear.bias is not None:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SIREN activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        if self.linear is not None:
            x = self.linear(x)

        if self.dropout is not None:
            x = self.dropout(x)

        return self.activation(x)

    def extra_repr(self) -> str:
        return f"w0={self.w0}, is_first={self.is_first}"


class SIRENNet(nn.Module):
    """Full SIREN network with proper layer structure.

    This builds a complete SIREN network with:
    - First layer with higher w0_initial
    - Hidden layers with standard w0
    - Optional final linear output layer
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        dropout: float = 0.0,
        degree_input: bool = False,
    ):
        """Initialize SIREN network.

        Args:
            in_features: Input dimension
            hidden_features: Hidden layer dimension
            out_features: Output dimension
            num_layers: Number of hidden layers
            w0: Frequency for hidden layers
            w0_initial: Frequency for first layer
            use_bias: Whether to use bias
            dropout: Dropout probability
            degree_input: Whether input is in degrees (will normalize to [-pi, pi])
        """
        super().__init__()
        self.degree_input = degree_input
        self.num_layers = num_layers

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            SIRENActivation(
                in_features=in_features,
                out_features=hidden_features,
                w0=w0,
                w0_initial=w0_initial,
                is_first=True,
                use_bias=use_bias,
                dropout=dropout,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                SIRENActivation(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    w0=w0,
                    is_first=False,
                    use_bias=use_bias,
                    dropout=dropout,
                )
            )

        # Output layer (linear, no activation)
        self.output = nn.Linear(hidden_features, out_features, bias=use_bias)
        self._init_output(hidden_features)

    def _init_output(self, in_features: int):
        """Initialize output layer weights."""
        w_std = math.sqrt(6.0 / in_features) / 1.0
        nn.init.uniform_(self.output.weight, -w_std, w_std)
        if self.output.bias is not None:
            nn.init.uniform_(self.output.bias, -w_std, w_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SIREN network.

        Args:
            x: Input tensor of shape (batch, in_features)

        Returns:
            Output tensor of shape (batch, out_features)
        """
        if self.degree_input:
            x = torch.deg2rad(x) - math.pi

        for layer in self.layers:
            x = layer(x)

        return self.output(x)
