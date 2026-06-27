from typing import Any, Optional
import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """
    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles

    Notes:
        This implementation is written for jagged nested tensors. Sequence
        boundaries come from the nested tensor offsets instead of padded tokens
        and padding masks.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10000,
        device=None,
        dtype=None
    ) -> None:
        """Create a RoPE module for nested attention heads.

        Args:
            dim: Per-head embedding dimension.
            max_seq_len: Largest position cached ahead of time.
            base: Base used to generate inverse frequencies.
            device: Optional device used for cached buffers.
            dtype: Unused placeholder kept for API compatibility.

        Returns:
            None.
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init(device)

    def rope_init(self, device=None):
        """Initialize the inverse-frequency buffer and the RoPE cache.

        Args:
            device: Optional device used to create cached tensors.

        Returns:
            None.
        """
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        """Precompute cosine and sine values up to ``max_seq_len``.

        Args:
            max_seq_len: Maximum absolute position stored in the cache.

        Returns:
            None.
        """
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Jagged nested tensor with shape
                ``[b, s, n_h, h_d]``.
            input_pos (Optional[torch.Tensor]): Optional nested tensor containing
                token position ids. During training, these ids describe each
                token position inside its own packed sample. During inference,
                they can provide the absolute position of the current token. If
                ``None``, positions are reconstructed from the nested tensor
                offsets rather than from padded indices.

        Returns:
            torch.Tensor: Nested tensor with shape ``[b, s, n_h, h_d]``.

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().unflatten(-1, [-1, 2])
        offsets = xshaped.offsets()
        seq_len = torch.diff(offsets)
        min_seqlen = torch.min(seq_len)
        max_seqlen = torch.max(seq_len)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        # extract the values based on whether input_pos is set or not
        if input_pos is None:
            # rope_cache = torch.nested.nested_tensor_from_jagged(
            #     torch.nested.as_nested_tensor(
            #         [self.cache[:l].unsqueeze(1).expand(-1, xshaped.size(2), -1, -1) for l in seq_len],
            #         layout=torch.jagged
            #     ).values(),
            #     offsets=offsets,
            #     min_seqlen=min_seqlen,
            #     max_seqlen=max_seqlen
            # )
            rope_cache_values = self.cache[(
                torch.arange(offsets[-1], device=xshaped.device) - torch.repeat_interleave(offsets[:-1], seq_len)
            )].unsqueeze(1).expand(-1, xshaped.size(2), -1, -1)
        else:
            # rope_cache = torch.nested.nested_tensor_from_jagged(
            #     torch.nested.as_nested_tensor(
            #         [self.cache[p].unsqueeze(1).expand(-1, xshaped.size(2), -1, -1) for p in input_pos.unbind()],
            #         layout=torch.jagged
            #     ).values(),
            #     offsets=offsets,
            #     min_seqlen=min_seqlen,
            #     max_seqlen=max_seqlen
            # )
            rope_cache_values = self.cache[input_pos.values()].unsqueeze(1).expand(-1, xshaped.size(2), -1, -1)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped_values = xshaped.values()
        x_out_values = torch.stack(
            [
                xshaped_values[..., 0] * rope_cache_values[..., 0]
                - xshaped_values[..., 1] * rope_cache_values[..., 1],
                xshaped_values[..., 1] * rope_cache_values[..., 0]
                + xshaped_values[..., 0] * rope_cache_values[..., 1],
            ],
            -1,
        )
        x_out = torch.nested.nested_tensor_from_jagged(
            x_out_values, offsets=offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen)

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
