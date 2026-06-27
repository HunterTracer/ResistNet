import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, Module
from torch import Tensor
import copy
from typing import Optional, Union, Callable, Any
from te_layer import TransformerEncoderLayer
from td_layer import TransformerDecoderLayer


# We use this for exact parity with the PyTorch implementation, having the same init
# for every layer might not be necessary.
def _get_clones(module, N):
    """Return ``N`` deep-copied layers wrapped in a ``ModuleList``."""
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    """Stack encoder layers that operate on batch-first nested tensors."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        """Create an encoder stack.

        Args:
            encoder_layer: Prototype encoder layer to clone.
            num_layers: Number of encoder layers in the stack.
            norm: Optional normalization applied after the final layer.

        Returns:
            None.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_input_pos: Optional[Tensor] = None, is_causal=False):
        """Encode a nested source sequence.

        Args:
            src: Source nested tensor with shape ``[batch, src_len, d_model]``.
            mask: Optional attention mask passed to each encoder layer.
            src_input_pos: Optional nested position ids aligned with ``src``.
                Positions describe per-sample token indices inside the jagged
                nested tensor instead of relying on padded tokens.
            is_causal: Whether each encoder self-attention block should apply a
                causal mask.

        Returns:
            Encoded nested tensor with the same nested structure as ``src``.
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_input_pos=src_input_pos, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """Stack decoder layers that consume nested target and memory tensors."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        """Create a decoder stack.

        Args:
            decoder_layer: Prototype decoder layer to clone.
            num_layers: Number of decoder layers in the stack.
            norm: Optional normalization applied after the final layer.

        Returns:
            None.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_input_pos: Optional[Tensor] = None,
        memory_input_pos: Optional[Tensor] = None,
        tgt_is_causal=False,
        memory_is_causal=False
    ):
        """Decode a nested target sequence against nested encoder memory.

        Args:
            tgt: Target nested tensor with shape ``[batch, tgt_len, d_model]``.
            memory: Encoder output nested tensor with shape
                ``[batch, src_len, d_model]``.
            tgt_mask: Optional self-attention mask for target tokens.
            memory_mask: Optional cross-attention mask between target and memory.
            tgt_input_pos: Optional nested position ids for ``tgt``.
            memory_input_pos: Optional nested position ids for ``memory``.
            tgt_is_causal: Whether decoder self-attention is causal.
            memory_is_causal: Whether decoder cross-attention should be treated
                as causal.

        Returns:
            Decoded nested tensor with the same nested structure as ``tgt``.
        """
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_input_pos=tgt_input_pos,
                memory_input_pos=memory_input_pos,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal
            )
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class Transformer(nn.Module):
    """Transformer built around batch-first nested tensors instead of padding."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
        pos_embeddings: Optional[nn.Module] = None,
        device=None,
        dtype=None,
    ) -> None:
        """Create a nested-tensor transformer.

        Args:
            d_model: Model feature width.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Hidden width of each feed-forward block.
            dropout: Dropout probability used in attention and FFN blocks.
            activation: Activation function or activation name.
            custom_encoder: Optional externally provided encoder module.
            custom_decoder: Optional externally provided decoder module.
            layer_norm_eps: Epsilon used by layer normalization.
            norm_first: Whether to apply pre-norm ordering.
            bias: Whether linear and normalization layers use bias terms.
            pos_embeddings: Optional positional embedding module shared by
                attention layers.
            device: Optional device passed to module parameters.
            dtype: Optional dtype passed to module parameters.

        Returns:
            None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                norm_first,
                bias,
                pos_embeddings=pos_embeddings,
                **factory_kwargs,
            )
            encoder_norm = nn.LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                norm_first,
                bias,
                pos_embeddings=pos_embeddings,
                **factory_kwargs,
            )
            decoder_norm = nn.LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        r"""Initialize projection weights with Xavier uniform values."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_input_pos: Optional[Tensor] = None,
        tgt_input_pos: Optional[Tensor] = None,
        memory_input_pos: Optional[Tensor] = None,
        src_is_causal: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False
    ) -> Tensor:
        """Run encoder-decoder inference on nested source and target tensors.

        Args:
            src: Source nested tensor with shape ``[batch, src_len, d_model]``.
            tgt: Target nested tensor with shape ``[batch, tgt_len, d_model]``.
            src_mask: Optional encoder self-attention mask.
            tgt_mask: Optional decoder self-attention mask.
            memory_mask: Optional decoder cross-attention mask.
            src_input_pos: Optional nested position ids for ``src``.
            tgt_input_pos: Optional nested position ids for ``tgt``.
            memory_input_pos: Optional nested position ids for encoder memory.
            src_is_causal: Whether encoder attention is causal.
            tgt_is_causal: Whether decoder self-attention is causal.
            memory_is_causal: Whether decoder cross-attention is causal.

        Returns:
            Decoder output nested tensor with shape ``[batch, tgt_len, d_model]``.

        Notes:
            Sequence lengths are represented by the nested tensor layout, so
            this implementation does not depend on padding tokens or
            ``key_padding_mask`` inputs.
        """
        memory = self.encoder(
            src,
            mask=src_mask,
            src_input_pos=src_input_pos,
            is_causal=src_is_causal
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_input_pos=tgt_input_pos,
            memory_input_pos=memory_input_pos,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal
        )
        return output
