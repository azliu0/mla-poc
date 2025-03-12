import torch.nn as nn
from .attention import MultiHeadAttention, MultiHeadLatentAttention
from typing import Optional
import torch


class ModelDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        num_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        d_kv_latent: Optional[int],
        use_cache: bool,
        device: torch.device,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ModelBlock(
                    d_model=d_model,
                    d_head=d_head,
                    num_heads=num_heads,
                    max_batch_size=max_batch_size,
                    max_seq_len=max_seq_len,
                    d_kv_latent=d_kv_latent,
                    use_cache=use_cache,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class ModelBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        max_batch_size: int,
        max_seq_len: int,
        d_kv_latent: Optional[int],
        use_cache: bool,
        device: torch.device,
    ):
        super().__init__()
        if d_kv_latent is None:
            self.attn = MultiHeadAttention(
                d_model=d_model,
                d_head=d_head,
                num_heads=num_heads,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.attn = MultiHeadLatentAttention(
                d_model=d_model,
                d_head=d_head,
                num_heads=num_heads,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                d_kv_latent=d_kv_latent,
                device=device,
            )
        self.use_cache = use_cache
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output: torch.Tensor = self.attn(x, use_cache=self.use_cache)
        x = self.norm1(x + attn_output)

        ffn_output: torch.Tensor = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
