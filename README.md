# Code for Paper "ResistNet: Quantifying solar energy reduction during hurricanes with deep learning"
This repository provides the PyTorch implementation of ResistNet, a Transformer-style encoder-decoder model for quantifying hurricane-induced solar energy reduction. The implementation supports variable-length sequences through jagged nested tensors.

## Main Files

- [attn_predictor.py](./attn_predictor.py): Attention-based prediction model
- [transformer.py](./transformer.py): encoder, decoder, and Transformer wrapper
- [te_layer.py](./te_layer.py): encoder layer
- [td_layer.py](./td_layer.py): decoder layer
- [mha.py](./mha.py): multi-head attention for nested tensors
- [position_embeddings.py](./position_embeddings.py): positional embeddings
- [swiglu.py](./swiglu.py): SwiGLU and FFN blocks

## Requirements

Use a recent PyTorch version with support for:

- `torch.nested`
- `layout=torch.jagged`
- nested-tensor operations used by SDPA in this project

## Minimal Example

```python
import torch
from attn_predictor import AttnPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src = torch.nested.as_nested_tensor(
    [torch.randn(5, 8, device=device), torch.randn(3, 8, device=device)],
    layout=torch.jagged,
)
tgt = torch.nested.as_nested_tensor(
    [torch.randn(4, 7, device=device), torch.randn(2, 7, device=device)],
    layout=torch.jagged,
)

src_pos = torch.nested.as_nested_tensor(
    [torch.arange(5, device=device), torch.arange(3, device=device)],
    layout=torch.jagged,
)
tgt_pos = torch.nested.as_nested_tensor(
    [torch.arange(4, device=device), torch.arange(2, device=device)],
    layout=torch.jagged,
)

model = AttnPredictor(
    encoder_input_dim=8,
    decoder_input_dim=7,
    output_dim=1,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_feedforward=512,
    device=device
).to(device)

output = model(
    src=src,
    tgt=tgt,
    src_is_causal=False,
    tgt_is_causal=True,
    memory_is_causal=False,
    src_input_pos=src_pos,
    tgt_input_pos=tgt_pos,
    memory_input_pos=src_pos
)
```

`output` is also a jagged nested tensor, and its layout follows `tgt`.
