"""Modular Location Encoder for geographic coordinate embedding.

The LocationEncoder combines:
1. Positional Encoding (SH, Raw, RFF, Cartesian3D)
2. Neural Network with configurable activation (ReLU, SIREN, Spline, RFF)

This is the core component for learning location embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal

from .encodings import get_encoding
from .activations import get_activation, ReLUActivation, SplineActivation


class LocationEncoder(nn.Module):
    """Modular location encoder with configurable encoding and activation.

    Architecture:
    1. Positional Encoding: (lon, lat) -> features
    2. Neural Network: features -> embedding

    The neural network consists of:
    - Input layer: encoding_dim -> hidden_dim
    - Hidden layers: hidden_dim -> hidden_dim (with activation)
    - Output layer: hidden_dim -> output_dim
    """

    def __init__(
        self,
        # Encoding config
        encoding_type: str = "spherical_harmonics",
        encoding_config: Optional[Dict[str, Any]] = None,
        # Network config
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        # Activation config
        activation_type: str = "relu",
        activation_config: Optional[Dict[str, Any]] = None,
        # Other
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        """Initialize location encoder.

        Args:
            encoding_type: Type of positional encoding
                ("spherical_harmonics", "sh", "raw", "rff", "cartesian3d")
            encoding_config: Config dict for encoding (e.g., {"legendre_polys": 10})
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            activation_type: Activation function ("relu", "siren", "spline", "rff")
            activation_config: Config dict for activation (e.g., {"n_knots": 15})
            dropout: Dropout probability
            use_bias: Whether to use bias in linear layers
        """
        super().__init__()

        # Store config
        self.encoding_type = encoding_type
        self.activation_type = activation_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initialize encoding
        encoding_config = encoding_config or {}
        self.encoding = get_encoding(encoding_type, **encoding_config)
        input_dim = self.encoding.output_dim

        # Initialize activation config
        activation_config = activation_config or {}

        # Build network layers
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
        self.activations.append(
            self._make_activation(activation_type, activation_config, is_first=True)
        )

        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
            self.activations.append(
                self._make_activation(activation_type, activation_config, is_first=False)
            )

        # Output layer (no activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=use_bias)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize weights
        self._init_weights()

    def _make_activation(
        self, activation_type: str, config: dict, is_first: bool
    ) -> nn.Module:
        """Create activation module.

        Args:
            activation_type: Type of activation
            config: Activation config
            is_first: Whether this is the first layer

        Returns:
            Activation module
        """
        config = config.copy()
        config["is_first"] = is_first
        return get_activation(activation_type, **config)

    def _init_weights(self):
        """Initialize network weights."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.kaiming_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode geographic coordinates.

        Args:
            coords: Tensor of shape (batch, 2) with (lon, lat) in degrees

        Returns:
            Embeddings of shape (batch, output_dim)
        """
        # Apply positional encoding
        x = self.encoding(coords)

        # Apply network layers
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            x = activation(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)

        return x

    def get_encoding_dim(self) -> int:
        """Get the positional encoding output dimension."""
        return self.encoding.output_dim


class RegressionHead(nn.Module):
    """Simple regression head for coordinate-based prediction tasks.

    Used for tasks like elevation prediction, population density, etc.
    """

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 1,
        hidden_dim: Optional[int] = None,
        activation: str = "relu",
    ):
        """Initialize regression head.

        Args:
            input_dim: Input dimension (from location encoder)
            output_dim: Output dimension (1 for scalar prediction)
            hidden_dim: Optional hidden layer dimension
            activation: Activation function
        """
        super().__init__()

        if hidden_dim is None:
            # Direct linear mapping
            self.head = nn.Linear(input_dim, output_dim)
        else:
            # One hidden layer
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else get_activation(activation),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply regression head.

        Args:
            x: Embeddings from location encoder

        Returns:
            Predictions
        """
        return self.head(x)


class LocationRegressor(nn.Module):
    """Combined location encoder + regression head.

    For tasks like elevation, population, temperature prediction.
    """

    def __init__(
        self,
        encoding_type: str = "spherical_harmonics",
        encoding_config: Optional[Dict] = None,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation_type: str = "relu",
        activation_config: Optional[Dict] = None,
        output_dim: int = 1,
    ):
        """Initialize location regressor.

        Args:
            encoding_type: Positional encoding type
            encoding_config: Encoding configuration
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            activation_type: Activation function type
            activation_config: Activation configuration
            output_dim: Output dimension (1 for scalar)
        """
        super().__init__()

        self.encoder = LocationEncoder(
            encoding_type=encoding_type,
            encoding_config=encoding_config,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Encoder output = hidden_dim
            num_layers=num_layers,
            activation_type=activation_type,
            activation_config=activation_config,
        )

        self.head = RegressionHead(
            input_dim=hidden_dim,
            output_dim=output_dim,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Predict from coordinates.

        Args:
            coords: (batch, 2) tensor of (lon, lat)

        Returns:
            (batch, output_dim) predictions
        """
        embeddings = self.encoder(coords)
        return self.head(embeddings).squeeze(-1)


def build_location_encoder_from_config(config: Dict[str, Any]) -> LocationEncoder:
    """Build a LocationEncoder from a config dictionary.

    Args:
        config: Configuration dictionary with keys:
            - encoding: {type, ...}
            - network: {hidden_dim, num_layers, ...}
            - activation: {type, ...}

    Returns:
        Configured LocationEncoder
    """
    encoding_config = config.get("encoding", {})
    encoding_type = encoding_config.pop("type", "spherical_harmonics")

    network_config = config.get("network", {})
    hidden_dim = network_config.get("hidden_dim", 256)
    output_dim = network_config.get("output_dim", 256)
    num_layers = network_config.get("num_layers", 3)
    dropout = network_config.get("dropout", 0.0)

    activation_config = config.get("activation", {})
    activation_type = activation_config.pop("type", "relu")

    return LocationEncoder(
        encoding_type=encoding_type,
        encoding_config=encoding_config,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        activation_type=activation_type,
        activation_config=activation_config,
        dropout=dropout,
    )
