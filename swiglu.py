import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.transformer import _get_activation_fn
from typing import Optional, Union, Callable


class PackedSwiGLU(nn.Module):
    """Packed SwiGLU projection used inside feed-forward blocks."""

    def __init__(
        self,
        dim,
        model_dim,
        device=None,
        dtype=None,
    ):
        """Create the packed SwiGLU projection.

        Args:
            dim: Input feature width.
            model_dim: Output feature width after gating.
            device: Optional device passed to module parameters.
            dtype: Optional dtype passed to module parameters.

        Returns:
            None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * model_dim, bias=False, **factory_kwargs)

    def forward(self, x):
        """Project inputs into gated SwiGLU activations.

        Args:
            x: Input tensor with shape ``[..., dim]``.

        Returns:
            Tensor with shape ``[..., model_dim]`` after SwiGLU gating.
        """
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return F.silu(x1) * x3


class FFN(nn.Module):
    """Standard two-layer feed-forward network."""

    def __init__(
        self,
        d_model,
        dim_feedforward,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """Create a two-layer feed-forward block.

        Args:
            d_model: Input and output feature width.
            dim_feedforward: Hidden feature width.
            dropout: Dropout probability between the two projections.
            activation: Activation function or activation name.
            bias: Whether linear layers use bias terms.
            device: Optional device passed to module parameters.
            dtype: Optional dtype passed to module parameters.

        Returns:
            None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x):
        """Apply the feed-forward network.

        Args:
            x: Input tensor with shape ``[..., d_model]``.

        Returns:
            Tensor with shape ``[..., d_model]``.
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class PackedSwiGLUFFN(nn.Module):
    """Feed-forward network that uses packed SwiGLU gating."""

    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of,
        dropout: float = 0.1,
        ffn_dim_multiplier=None,
        device=None,
        dtype=None,
    ):
        """Create a SwiGLU-based feed-forward block.

        Args:
            dim: Input and output feature width.
            hidden_dim: Base hidden width before SwiGLU packing.
            multiple_of: Round the hidden width up to this multiple.
            dropout: Dropout probability before the output projection.
            ffn_dim_multiplier: Optional extra multiplier applied to the hidden
                width before rounding.
            device: Optional device passed to module parameters.
            dtype: Optional dtype passed to module parameters.

        Returns:
            None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Apply the packed SwiGLU feed-forward network.

        Args:
            x: Input tensor with shape ``[..., dim]``.

        Returns:
            Tensor with shape ``[..., dim]`` after projection back to model size.
        """
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(self.dropout(F.silu(x1) * x3))
