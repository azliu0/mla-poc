import torch
import torch.nn as nn
from .decoder import ModelDecoder
from typing import Optional
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device: torch.device):
        super().__init__()

        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(
            1
        )
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model),
        ).to(device)

        pe[:, 0::2] = torch.sin(position * div_term).to(device)
        pe[:, 1::2] = torch.cos(position * div_term).to(device)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_idx: Optional[int] = None) -> torch.Tensor:
        if start_idx is not None:
            return x + self.pe[:, start_idx : x.size(1) + start_idx, :]
        else:
            return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_head: int,
        num_heads: int,
        num_layers: int,
        max_seq_length: int,
        max_batch_size: int,
        d_kv_latent: Optional[int],
        use_cache: bool,
        device: torch.device,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model=d_model, max_len=max_seq_length, device=device
        )

        self.decoder = ModelDecoder(
            d_model=d_model,
            d_head=d_head,
            num_heads=num_heads,
            num_layers=num_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_length,
            d_kv_latent=d_kv_latent,
            use_cache=use_cache,
            device=device,
        )
        self.use_cache = use_cache
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, start_idx: Optional[int] = None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_encoding(x, start_idx)
        x = self.decoder(x)
        logits = self.output_projection(x)
        return logits
