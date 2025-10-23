import math
from typing import Optional, Set, Tuple, Literal
import numpy as np
import torch
from torch import nn
from models.time_embedding import get_timestep_embedding


class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        in_dim: Input layer dimension
        num_layers: Number of network layers
        layer_width: Width of each MLP layer
        out_dim: Output layer dimension. Uses layer_width if None.
        activation: intermediate layer activation function.
        out_activation: output activation function.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int,
        layer_width: int,
        out_dim: Optional[int] = None,
        skip_connections: Optional[Tuple[int]] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        out_activation: Optional[nn.Module] = None,
        use_time_embedding: bool = False,
        time_embedding_dim: int = 64,
        normal_zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        assert self.in_dim > 0
        self.out_dim = out_dim if out_dim is not None else layer_width
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.skip_connections = skip_connections
        self._skip_connections: Set[int] = set(skip_connections) if skip_connections else set()
        self.activation = activation
        self.out_activation = out_activation
        self.net = None

        self.use_time_embedding = use_time_embedding
        self.time_embedding_dim = time_embedding_dim

        self.build_nn_modules()

        if normal_zero_init:
            self.normal_zero_initialization()

    def build_nn_modules(self) -> None:
        """Initialize multi-layer perceptron."""
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            time_emb = self.time_embedding_dim if self.use_time_embedding else 0
            for i in range(self.num_layers - 1):
                if i == 0:
                    assert i not in self._skip_connections, "Skip connection at layer 0 doesn't make sense."
                    layers.append(nn.Linear(self.in_dim + time_emb, self.layer_width))
                elif i in self._skip_connections:
                    layers.append(nn.Linear(self.layer_width + self.in_dim + time_emb, self.layer_width))
                else:
                    layers.append(nn.Linear(self.layer_width, self.layer_width))
            layers.append(nn.Linear(self.layer_width, self.out_dim))
        self.layers = nn.ModuleList(layers)

        if self.use_time_embedding:
            self.temb = nn.Sequential(
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
                nn.ReLU(),
                nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            )

    def normal_zero_initialization(self) -> None:
        """Initialize network weights."""
        for layer in self.layers:
            # n_dim = layer.in_features
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, in_tensor: torch.Tensor, timestamp=None) -> torch.Tensor:
        """Process input with a multilayer perceptron.

        Args:
            in_tensor: Network input. shape: ("*bs in_dim")

        Returns:
            MLP network output. shape: ("*bs out_dim")
        """

        if self.use_time_embedding:
            assert timestamp is not None
            time_emb = get_timestep_embedding(timestamp, self.time_embedding_dim)
            time_emb = self.temb(time_emb)
            # in_tensor is (..., in_dim)
            # time_emb is (1, time_embedding_dim)
            # we need to broadcast time_emb to match the shape of in_tensor
            time_emb = time_emb.expand(in_tensor.shape[:-1] + (-1,))
            in_tensor = torch.cat([in_tensor, time_emb], dim=-1)
        else:
            time_emb = None

        x = in_tensor
        for i, layer in enumerate(self.layers):
            # as checked in `build_nn_modules`, 0 should not be in `_skip_connections`
            if i in self._skip_connections:
                x = torch.cat([in_tensor, x], -1)
            x = layer(x)
            if self.activation is not None and i < len(self.layers) - 1:
                x = self.activation(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x
