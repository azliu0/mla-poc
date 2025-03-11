# mla-poc

reference: https://arxiv.org/pdf/2405.04434

## Usage

this repo contains four separate modules. each of these modules implements and benchmarks MLA against KV-cache with standard multi-head attention, reproducing the memory and latency results from the deepseek paper.

- `pytorch`: reference implementation in PyTorch
- `jax`: reference implementation in JAX
- `cuda`: using cuda kernels
- `triton`: using triton kernels

## Architecture

TODO: architecture, num heads, num params

## Benchmarks

TODO

add something about running on the same inputs:

- according to https://www.thonking.ai/p/strangely-matrix-multiplications, the inputs matter
