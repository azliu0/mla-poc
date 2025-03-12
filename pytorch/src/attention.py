import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class Cache:
    @abstractmethod
    def update(self, x: torch.Tensor, position: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def get(self, batch_size: int) -> torch.Tensor:
        pass


class KVCache(Cache):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.max_seq_len = max_seq_len
        self.cur_seq_len = 0

        self.k_cache = torch.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), device=device
        )
        self.v_cache = torch.zeros(
            (max_batch_size, max_seq_len, num_heads, head_dim), device=device
        )

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor, position: Optional[int] = None
    ) -> None:
        batch_size, seq_len, _, _ = k_new.shape

        if position is None:
            position = self.cur_seq_len

        self.k_cache[:batch_size, position : position + seq_len] = k_new
        self.v_cache[:batch_size, position : position + seq_len] = v_new

        if position + seq_len > self.cur_seq_len:
            self.cur_seq_len = position + seq_len

    def get(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k_cache[:batch_size, : self.cur_seq_len],
            self.v_cache[:batch_size, : self.cur_seq_len],
        )


class KVLatentCache(Cache):
    # The main difference for KVLatentCache is that we only need to store the latent values
    # The upsampled keys and values are computed later
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        d_kv_latent: int,
        device: torch.device,
    ):
        self.max_seq_len = max_seq_len
        self.cur_seq_len = 0

        self.latent_cache = torch.zeros(
            (max_batch_size, max_seq_len, d_kv_latent), device=device
        )

    def update(self, latent: torch.Tensor, position: Optional[int] = None) -> None:
        batch_size, seq_len, _ = latent.shape

        if position is None:
            position = self.cur_seq_len

        self.latent_cache[:batch_size, position : position + seq_len] = latent

        if position + seq_len > self.cur_seq_len:
            self.cur_seq_len = position + seq_len

    def get(self, batch_size: int) -> torch.Tensor:
        return self.latent_cache[:batch_size, : self.cur_seq_len]


class BaseAttention(nn.Module, ABC):
    @abstractmethod
    def forward_no_cache(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward_with_cache(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        pass


class MultiHeadAttention(BaseAttention):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_head * num_heads)
        self.k_proj = nn.Linear(d_model, d_head * num_heads)
        self.v_proj = nn.Linear(d_model, d_head * num_heads)
        self.out_proj = nn.Linear(d_head * num_heads, d_model)

        self.device = device
        self.kv_cache = None

    def forward_no_cache(self, x: torch.Tensor) -> torch.Tensor:
        # assumption: x represents the entire sequence
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)

        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(context)

        return output

    def forward_with_cache(self, x: torch.Tensor) -> torch.Tensor:
        # assumption: x represents new tokens, and the previous tokens are cached
        batch_size, seq_len, _ = x.shape

        if self.kv_cache is None:
            self.kv_cache = KVCache(
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                num_heads=self.num_heads,
                head_dim=self.d_head,
                device=self.device,
            )

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)

        k_new = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)
        v_new = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head)

        self.kv_cache.update(k_new, v_new, position=None)

        k, v = self.kv_cache.get(batch_size)

        # (b, n_h, s_x, d_h)
        q = q.transpose(1, 2)
        # (b, n_h, s_t, d_h)
        k = k.transpose(1, 2)
        # (b, n_h, s_t, d_h)
        v = v.transpose(1, 2)

        # (b, n_h, s_x, s_t)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # (s_x, s_t)
        total_seq_len = k.size(-2)
        causal_mask = torch.triu(
            torch.ones(seq_len, total_seq_len, device=self.device, dtype=torch.bool),
            diagonal=1 + total_seq_len - seq_len,
        )

        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_head * self.num_heads)
        )
        output = self.out_proj(context)

        return output

    def forward(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        if use_cache:
            return self.forward_with_cache(x)
        else:
            return self.forward_no_cache(x)


class MultiHeadLatentAttention(BaseAttention):
    def __init__(
        self,
        d_model: int,
        d_head: int,
        num_heads: int,
        max_batch_size: int,
        max_seq_len: int,
        d_kv_latent: int,
        device: torch.device,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.d_kv_latent = d_kv_latent
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        self.downsample = nn.Linear(d_model, d_kv_latent, bias=False)

        self.upsample_key = nn.Linear(d_kv_latent, d_head * num_heads, bias=False)
        self.upsample_value = nn.Linear(d_kv_latent, d_head * num_heads, bias=False)

        self.q_proj = nn.Linear(d_model, d_head * num_heads, bias=False)
        self.out_proj = nn.Linear(d_head * num_heads, d_model, bias=False)

        self.device = device

        # Invariant: all three of these are None or populated
        self.latent_cache = None
        self.q_uk_combined = None
        self.out_uv_combined = None

    def forward_no_cache(self, x: torch.Tensor) -> torch.Tensor:
        # assumption: x represents the entire sequence
        batch_size, seq_len, _ = x.shape

        latents = self.downsample(x)

        keys = self.upsample_key(latents)
        values = self.upsample_value(latents)

        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        k = keys.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = values.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(
            1, 2
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_head * self.num_heads)
        )

        output = self.out_proj(context)

        return output

    def forward_with_cache(self, x: torch.Tensor) -> torch.Tensor:
        # assumption: x represents new tokens, and the previous tokens are cached
        batch_size, seq_len, _ = x.shape

        if self.latent_cache is None:
            self.latent_cache = KVLatentCache(
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                d_kv_latent=self.d_kv_latent,
                device=self.device,
            )

            # cache W^Q^T @ W_UK over the head dimension to avoid recomputation
            # (d_h * n_h, d_m) -> (n_h, d_h, d_m)
            q_weight = self.q_proj.weight.view(
                self.num_heads, self.d_head, self.d_model
            )
            # (n_h, d_h, d_m) -> (n_h, d_m, d_h)
            q_weight = q_weight.transpose(1, 2)

            # (d_h * n_h, d_c) -> (n_h, d_h, d_c)
            uk_weight = self.upsample_key.weight.view(
                self.num_heads, self.d_head, self.d_kv_latent
            )

            # (n_h, d_m, d_h) * (n_h, d_h, d_c) -> (n_h, d_m, d_c)
            self.q_uk_combined = torch.matmul(q_weight, uk_weight)

            # cache W_O @ W_UV over the head dimension to avoid recomputation
            # (d_m, d_h * n_h) -> (d_m, n_h, d_h)
            o_weight = self.out_proj.weight.view(
                self.d_model, self.num_heads, self.d_head
            )
            # (d_m, n_h, d_h) -> (n_h, d_m, d_h)
            o_weight = o_weight.transpose(0, 1)

            # (d_h * n_h, d_c) -> (n_h, d_h, d_c)
            uv_weight = self.upsample_value.weight.view(
                self.num_heads, self.d_head, self.d_kv_latent
            )

            # (n_h, d_m, d_h) * (n_h, d_h, d_c) -> (n_h, d_m, d_c)
            self.out_uv_combined = torch.matmul(o_weight, uv_weight)

        assert self.latent_cache is not None
        assert self.q_uk_combined is not None
        assert self.out_uv_combined is not None

        # (b, s_x, d_m) -> (b, s_x, d_c)
        new_latents = self.downsample(x)

        self.latent_cache.update(new_latents, position=None)

        # (b, s_t, d_c)
        all_latents = self.latent_cache.get(batch_size)
        # (b, s_t, d_c) -> (b, 1, s_t, d_c) -> (b, n_h, s_t, d_c)
        all_latents_expanded = all_latents.unsqueeze(1).expand(
            -1, self.num_heads, -1, -1
        )

        # (b, s_x, d_m) -> (b, 1, s_x, d_m) -> (b, n_h, s_x, d_m)
        x_expanded = x.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # (b, n_h, s_x, d_m) * (1, n_h, d_m, d_c) -> (b, n_h, s_x, d_c)
        query_part = torch.matmul(x_expanded, self.q_uk_combined.unsqueeze(0))

        # (n_h, d_m, d_c) * (b, n_h, d_c, s_t) -> (b, n_h, d_m, s_t)
        value_part = torch.matmul(
            self.out_uv_combined, all_latents_expanded.transpose(2, 3)
        )
        # (b, n_h, d_m, s_t) -> (b, n_h, s_t, d_m)
        value_part = value_part.transpose(2, 3)

        # query_part: (b, n_h, s_x, d_c)
        # all_latents: (b, s_t, d_c) -> (b, 1, s_t, d_c) -> (b, 1, d_c, s_t)
        # (b, n_h, s_x, d_c) * (b, 1, d_c, s_t) -> (b, n_h, s_x, s_t)
        scores = torch.matmul(
            query_part, all_latents.unsqueeze(1).transpose(2, 3)
        ) / math.sqrt(self.d_head)

        # (s_x, s_t)
        total_seq_len = all_latents.size(-2)
        causal_mask = torch.triu(
            torch.ones(seq_len, total_seq_len, device=self.device, dtype=torch.bool),
            diagonal=1 + total_seq_len - seq_len,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # (b, n_h, s_x, s_t) softmaxed over the entire sequence
        attn_weights = F.softmax(scores, dim=-1)

        # (b, n_h, s_x, s_t) * (b, n_h, s_t, d_m) -> (b, n_h, s_x, d_m)
        output = torch.matmul(attn_weights, value_part)

        # (b, n_h, s_x, d_m) -> (b, s_x, n_h, d_m)
        output = output.transpose(1, 2)

        # (b, s_x, n_h, d_m) -> (b, s_x, d_m)
        output = output.sum(dim=2)

        return output

    def forward(self, x: torch.Tensor, use_cache: bool) -> torch.Tensor:
        if use_cache:
            return self.forward_with_cache(x)
        else:
            return self.forward_no_cache(x)
