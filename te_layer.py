import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mha import MultiHeadAttention
from swiglu import PackedSwiGLUFFN, FFN
from typing import Optional, Union, Callable



class TransformerEncoderLayer(nn.Module):
    """Single encoder block for batch-first jagged nested tensors."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
        pos_embeddings: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ) -> None:
        """Create one encoder layer.

        Args:
            d_model: Model feature width.
            nhead: Number of attention heads.
            dim_feedforward: Hidden width of the feed-forward block.
            dropout: Dropout probability used after attention and FFN blocks.
            activation: Activation function or activation name.
            layer_norm_eps: Epsilon used by layer normalization.
            norm_first: Whether to apply pre-norm ordering.
            bias: Whether linear and normalization layers use bias terms.
            pos_embeddings: Optional positional embedding module applied to
                query and key tensors.
            device: Optional device passed to module parameters.
            dtype: Optional dtype passed to module parameters.

        Returns:
            None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            pos_embeddings=pos_embeddings,
            **factory_kwargs,
        )

        # Legacy string support for activation function.
        if isinstance(activation, str) and activation == 'swiglu':
            self.ffn = PackedSwiGLUFFN(dim=d_model, hidden_dim=dim_feedforward, multiple_of=128, dropout=dropout,
                                       **factory_kwargs)
        else:
            self.ffn = FFN(d_model=d_model, dim_feedforward=dim_feedforward,
                           dropout=dropout, activation=activation, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        is_causal: bool = False
    ) -> Tensor:
        """Apply self-attention to a nested sequence.

        Args:
            x: Nested tensor with shape ``[batch, seq_len, d_model]``.
            attn_mask: Optional attention mask broadcastable to SDPA.
            input_pos: Optional nested position ids aligned with ``x``.
            is_causal: Whether to apply causal attention.

        Returns:
            Attention output with the same nested structure as ``x``.
        """
        x = self.self_attn(x, x, x, attn_mask=attn_mask,
                           query_input_pos=input_pos, key_input_pos=input_pos, is_causal=is_causal)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network to a nested sequence.

        Args:
            x: Nested tensor with shape ``[batch, seq_len, d_model]``.

        Returns:
            Feed-forward output with the same nested structure as ``x``.
        """
        return self.dropout2(self.ffn(x))

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_input_pos: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Run one encoder layer over a nested source sequence.

        Args:
            src: Source nested tensor with shape ``[batch, seq_len, d_model]``.
            src_mask: Optional self-attention mask for ``src``.
            src_input_pos: Optional nested position ids for ``src``. These
                positions are used with jagged nested tensors rather than
                deriving validity from padded tokens.
            is_causal: Whether self-attention should be causal.

        Returns:
            Updated nested tensor with the same shape and layout as ``src``.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask=src_mask, input_pos=src_input_pos, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask=src_mask, input_pos=src_input_pos, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
