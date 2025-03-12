# mla-poc

reference: https://arxiv.org/pdf/2405.04434

more detailed implementation reference and results [here](details.md)

## benchmarks

see model parameters [here](https://github.com/azliu0/mla-poc/blob/main/config.yml). all benchmarks on a 7B architecture which is roughly deepseek-llm-7b-base (adjusted to support mla).

on a single gpu `NVIDIA A100 80GB PCIe` running on [Modal Labs](https://modal.com/):

| model | latency | memory | flops | kv-cache size |
|-------|-------------|------------|-----------|---------------|
| no-cache | 999.01 ms | 26.73 GiB | 1364.08B | n/a |
| kv-cache | 40.44 ms | 27.94 GiB | 33.87B | 1920.0 MiB |
| mla+kv-cache | 36.24 ms | 26.16 GiB | 35.52B | 30.0 MiB |

- kv-cache size reduction: 98.4%
- mla compute overhead (theory): ~2.4B flops
- mla compute overhead (practice): 1.65B flops

## reproducibility

`python3 -m src.inference --benchmark`, you'll need a modal account to run the script

`python3 -m src.inference --no-mla-correctness --mla-correctness` to assert correctness of models with and without cache
