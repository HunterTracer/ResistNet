import torch
from torch import Tensor
from torch import nn
from te_layer import TransformerEncoderLayer
from td_layer import TransformerDecoderLayer
from transformer import TransformerEncoder, TransformerDecoder
from position_embeddings import RotaryPositionalEmbeddings
from typing import Optional


class AttnPredictor(nn.Module):
    """Encoder-decoder predictor for jagged nested input sequences."""

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_input_dim: int,
        output_dim: int = 1,
        max_seq_len: int = 1024,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        input_proj: str = 'linear',
        activation: str = 'swiglu',
        output_proj: str = 'sigmoid',
        pos_embed: str = 'rope',
        norm_first: bool = False,
        device=None,
        dtype=None
    ) -> None:
        """Create the attention-based predictor.

        Args:
            encoder_input_dim: Feature width of encoder inputs.
            decoder_input_dim: Feature width of decoder inputs.
            output_dim: Feature width of decoder predictions.
            max_seq_len: Maximum position cached by positional embeddings.
            d_model: Shared transformer feature width.
            nhead: Number of attention heads.
            num_layers: Number of encoder and decoder layers.
            dim_feedforward: Hidden width of feed-forward blocks.
            dropout: Dropout probability used across the model.
            input_proj: Input projection type for source and target features.
            activation: Feed-forward activation name.
            output_proj: Output projection type for predictions.
            pos_embed: Positional embedding type.
            norm_first: Whether transformer layers use pre-norm ordering.
            device: Optional device passed to module parameters.
            dtype: Optional dtype passed to module parameters.

        Returns:
            None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.head_dim = d_model // nhead

        if input_proj == 'linear':
            self.encoder_input_proj = nn.Linear(encoder_input_dim, d_model, **factory_kwargs)
            self.decoder_input_proj = nn.Linear(decoder_input_dim, d_model, **factory_kwargs)
        else:
            raise NotImplementedError

        if pos_embed == 'rope':
            self.pos_embeddings = RotaryPositionalEmbeddings(self.head_dim, max_seq_len, **factory_kwargs)
        else:
            self.pos_embeddings = None

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            pos_embeddings=self.pos_embeddings,
            **factory_kwargs
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            pos_embeddings=self.pos_embeddings,
            **factory_kwargs
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)

        if output_proj == 'linear':
            self.decoder_output_proj = nn.Linear(d_model, output_dim, **factory_kwargs)
        elif output_proj == 'relu':
            self.decoder_output_proj = nn.Sequential(
                nn.Linear(d_model, output_dim, **factory_kwargs),
                nn.ReLU()
            )
        elif output_proj == 'sigmoid':
            self.decoder_output_proj = nn.Sequential(
                nn.Linear(d_model, output_dim, **factory_kwargs),
                nn.Sigmoid()
            )
        else:
            raise NotImplementedError

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_is_causal: bool,
        tgt_is_causal: bool,
        memory_is_causal: bool,
        src_input_pos: Optional[Tensor] = None,
        tgt_input_pos: Optional[Tensor] = None,
        memory_input_pos: Optional[Tensor] = None
    ) -> Tensor:
        """Predict decoder outputs from nested source and target sequences.

        Args:
            src: Source nested tensor with shape
                ``[batch, src_len, encoder_input_dim]``.
            tgt: Target nested tensor with shape
                ``[batch, tgt_len, decoder_input_dim]``.
            src_is_causal: Whether encoder self-attention is causal.
            tgt_is_causal: Whether decoder self-attention is causal.
            memory_is_causal: Whether decoder cross-attention is causal.
            src_input_pos: Optional nested position ids for ``src``.
            tgt_input_pos: Optional nested position ids for ``tgt``.
            memory_input_pos: Optional nested position ids for encoder memory.

        Returns:
            Nested tensor with shape ``[batch, tgt_len, output_dim]``.

        Notes:
            Variable-length samples are represented with nested tensors
            throughout the model, so no padding-based masking path is assumed.
        """
        src = self.encoder_input_proj(src)  # (B, jL, d_model)
        tgt = self.decoder_input_proj(tgt)  # (B, jL, d_model)

        memory = self.transformer_encoder(src, src_input_pos=src_input_pos, is_causal=src_is_causal)  # (B, jL, d_model)
        output = self.transformer_decoder(
            tgt, memory,
            tgt_input_pos=tgt_input_pos, memory_input_pos=memory_input_pos,
            tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal
        )

        output = self.decoder_output_proj(output)  # (B, jL, 1)
        return output


if __name__ == "__main__":
    device = torch.device('cuda')
    model = AttnPredictor(encoder_input_dim=8, decoder_input_dim=7, output_dim=1, device=device)
    model.compile()
