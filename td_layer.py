import torch
import torch.nn as nn
import torch.nn.functional as F
from mha import MultiHeadAttention
from swiglu import PackedSwiGLUFFN, FFN
from torch import Tensor
from typing import Optional, Union, Callable


class TransformerDecoderLayer(nn.Module):
    """Single decoder block for batch-first jagged nested tensors."""

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
        """Create one decoder layer.

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
        self.multihead_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            pos_embeddings=pos_embeddings,
            **factory_kwargs,
        )

        if isinstance(activation, str) and activation == 'swiglu':
            self.ffn = PackedSwiGLUFFN(dim=d_model, hidden_dim=dim_feedforward, multiple_of=128, dropout=dropout,
                                       **factory_kwargs)
        else:
            self.ffn = FFN(d_model=d_model, dim_feedforward=dim_feedforward,
                           dropout=dropout, activation=activation, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        input_pos: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        """Apply decoder self-attention to the nested target sequence.

        Args:
            x: Target nested tensor with shape ``[batch, tgt_len, d_model]``.
            attn_mask: Optional target self-attention mask.
            input_pos: Optional nested target position ids.
            is_causal: Whether to apply causal self-attention.

        Returns:
            Self-attention output with the same nested structure as ``x``.
        """
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_input_pos=input_pos,
            key_input_pos=input_pos,
            is_causal=is_causal
        )
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        tgt_input_pos: Optional[Tensor],
        memory_input_pos: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        """Apply cross-attention from target tokens to encoder memory.

        Args:
            x: Target nested tensor with shape ``[batch, tgt_len, d_model]``.
            mem: Memory nested tensor with shape ``[batch, src_len, d_model]``.
            attn_mask: Optional cross-attention mask.
            tgt_input_pos: Optional nested position ids for target tokens.
            memory_input_pos: Optional nested position ids for memory tokens.
            is_causal: Whether to apply causal cross-attention.

        Returns:
            Cross-attention output with the same nested structure as ``x``.
        """
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            query_input_pos=tgt_input_pos,
            key_input_pos=memory_input_pos,
            is_causal=is_causal
        )
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        """Apply the feed-forward network to the nested target sequence.

        Args:
            x: Nested tensor with shape ``[batch, tgt_len, d_model]``.

        Returns:
            Feed-forward output with the same nested structure as ``x``.
        """
        return self.dropout3(self.ffn(x))

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_input_pos: Optional[Tensor] = None,
        memory_input_pos: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ) -> Tensor:
        """Run one decoder layer on nested target and memory tensors.

        Args:
            tgt: Target nested tensor with shape ``[batch, tgt_len, d_model]``.
            memory: Encoder memory nested tensor with shape
                ``[batch, src_len, d_model]``.
            tgt_mask: Optional target self-attention mask.
            memory_mask: Optional cross-attention mask.
            tgt_input_pos: Optional nested position ids for ``tgt``.
            memory_input_pos: Optional nested position ids for ``memory``.
            tgt_is_causal: Whether decoder self-attention is causal.
            memory_is_causal: Whether decoder cross-attention is causal.

        Returns:
            Updated nested tensor with the same shape and layout as ``tgt``.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_input_pos, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                tgt_input_pos,
                memory_input_pos,
                memory_is_causal
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_input_pos, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, tgt_input_pos, memory_input_pos, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x
