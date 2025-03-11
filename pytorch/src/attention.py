import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads

        self.q_proj = nn.Linear(d_model, d_head * num_heads)
        self.k_proj = nn.Linear(d_model, d_head * num_heads)
        self.v_proj = nn.Linear(d_model, d_head * num_heads)
        self.out_proj = nn.Linear(d_head * num_heads, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        # TODO: implement caching
        k = (
            self.k_proj(x)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_head)
        )
        output = self.out_proj(context)

        return output


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, d_head: int, num_heads: int, d_kv_latent: int):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.d_kv_latent = d_kv_latent
        self.num_heads = num_heads

        self.downsample = nn.Linear(d_model, d_kv_latent)

        self.upsample_key = nn.Linear(d_kv_latent, d_model)
        self.upsample_value = nn.Linear(d_kv_latent, d_model)

        self.q_proj = nn.Linear(d_model, d_head * num_heads)
        self.out_proj = nn.Linear(d_head * num_heads, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # TODO: implement caching
        latents = self.downsample(x)

        keys = self.upsample_key(latents)
        values = self.upsample_value(latents)

        q = (
            self.q_proj(x)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        k = keys.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = values.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_head * self.num_heads)
        )

        output = self.out_proj(context)

        return output
