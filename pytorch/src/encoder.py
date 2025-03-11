import torch.nn as nn
from .attention import MultiHeadAttention, MultiHeadLatentAttention
from typing import Optional


class ModelEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        num_layers: int,
        d_kv_latent: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ModelBlock(d_model, d_head, num_heads, d_kv_latent)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)


class ModelBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        d_kv_latent: Optional[int] = None,
    ):
        super().__init__()
        if d_kv_latent is None:
            self.attn = MultiHeadAttention(d_model, d_head, num_heads)
        else:
            self.attn = MultiHeadLatentAttention(
                d_model, d_head, num_heads, d_kv_latent
            )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x
