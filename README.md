# mla-poc

reference: https://arxiv.org/pdf/2405.04434

details: [details](details.md)

| model | latency | memory | flops | kv-cache size |
|-------|-------------|------------|-----------|---------------|
| no-cache | 999.01 ms | 26.73 GiB | n/a | n/a |
| kv-cache | 40.44 ms | 27.94 GiB | 33.87B | 1920.0 MiB |
| mla+kv-cache | 36.24 ms | 26.16 GiB | 35.52B | 30.0 MiB |

kv-cache size reduction: 98.4%

mla compute overhead (theory): ~2.4B flops

mla compute overhead (practice): 1.65B flops
