# details

## implementation

from the paper: "$W^{UK}$ can be absorbed into $W^Q$ and $W^{UV}$ can be absorbed into $W^O$", which is the crux of the implementation.

```math
\begin{align*}
q_{ti}^Tk_{jk} &= (W_Q^ih_t)^T(W^{UKi}W^{DKVi}h_j) \\
&= h_t^T(W_{Q}^{Ti}W^{UKi})(W^{DKVi}h_j) \\
&= h_t^T\tilde{W}_Q^i c_{tj}
\end{align*}
```

$c_{tj}$ is cached and we can precompute $\tilde{W}\_Q$. the index $i$ is the head index, i.e., $W_Q^i = W_Q[i * d_h : (i + 1) * d_h, :]$, so in practice we have to reshape $W_Q \rightarrow (n_h, d_h, d_m)$ and $W^{UK} \rightarrow (n_h, d_h, d\_{kv})$ before computing their product.

as a sanity check, we also have to make sure that the computation at inference-time is flop efficient, i.e. roughly the same as normal attention.

since we're computing the product over the entire token space, we expect this matrix multiply to look like this:

```math
\begin{align*}
(b, n_h, s_x, d_m) \times (b, n_h, d_m, d_c) \times (b, n_h, d_c, s_t) \\ \rightarrow (b, n_h, s_x, s_t)
\end{align*}
```

where $s_x$ is the length of the incoming tokens and $s_t$ is the total sequence length.

what we want to avoid is the amount of computation that we do in the $s_t$ dimension, since this is the largest dimension; therefore, we should implement this as a "query" multiply (first two matrices) and then a "score" multiply (third matrix).

absorbing $W^{UV}$ into $W^O$:

```math
\begin{align*}
W_O[u_{t,1}, \ldots, u_{t,s_t}] = \sum_{i=1}^{n_h}\sum_{j=1}^{s_t} \alpha_{t,i,j} W_O^iW^{UVi}c_{j,i}
\end{align*}
```

so we should precompute $\tilde{W}\_O = [W_O^i W^{UVi}]\_{i=1}^{n_h}$ which has size $(n_h, d_m, d_c)$.

At inference-time, it's tempting to do the following:
- compute scores $(h_t^T\tilde{W}_q)c$, which follows shape $(b, n_h, s_x, d_c) \times (b, n_h, d_c, s_t) \rightarrow (b, n_h, s_x, s_t)$
- compute "value part" $\tilde{W}_Oc$, which follows shape $(b, n_h, d_m, d_c) \times (b, n_h, d_c, s_t) \rightarrow (b, n_h, d_m, s_t)$
- compute the final output vector: $\text{scores} \times \text{value part}$, which follows shape $(b, n_h, s_x, s_t) \times (b, n_h, d_m, s_t) \rightarrow (b, n_h, s_x, d_m)$

this is flop inefficient, since the second step multiplies $s_t$ and $d_m$ together, which is the largest dimension and something that normal attention avoids.

instead, we need to rearrange the computation to first eliminate $s_t$ before $\tilde{W}_Oc$. this can be done with a bit of linear algebra:

```math
\begin{align*}
u_t^i[k] &= \sum_{j=1}^{s_t} \alpha_{t,i,j}(\tilde{W}_O^ic_{j,i})[k] \\
&= \sum_{j=1}^{s_t} \alpha_{t,i,j}\left(\sum_{l=1}^{d_c} \tilde{W}_O^i[k,l]c_{j,i}[l]\right) \\
&= \sum_{l=1}^{d_c} \tilde{W}_O^i[k,l]\left(\sum_{j=1}^{s_t} \alpha_{t,i,j}c_{j,i}[l]\right) \\
\end{align*}
```

In other words,
```math
u_t^i = \tilde{W}_O^i \sum_{j=1}^{s_t} \alpha_{t,i,j}c_{j,i}
```

so our inference-time implementation is:
- (same as before) compute scores $(h_t^T\tilde{W}_q)c$, which follows shape $(b, n_h, s_x, d_c) \times (b, n_h, d_c, s_t) \rightarrow (b, n_h, s_x, s_t)$
- multiply attention scores by latents, i.e., $Z = \alpha\cdot c$: $(b, n_h, s_x, s_t) \times (b, n_h, s_t, d_c) \rightarrow (b, n_h, s_x, d_c)$
- project to output space, i.e., $\text{output} = Z\tilde{W}_O^T$: $(b, n_h, s_x, d_c) \times (b, n_h, d_c, d_m) \rightarrow (b, n_h, s_x, d_m)$

now, all matrix multiplies on the axis $s_t$ use dimensions that are strictly smaller than $d_m$, so this implementation is flop efficient. in the analysis below, we show that it is now essentially the same as normal attention.

## results

- slow mla implementation, i.e. without rearranging the computation: https://github.com/azliu0/mla-poc/blob/f2b16b00a5cf2f73c7c4d9af27ee47a99998f9b6/pytorch/src/attention.py#L370
- fast mla implementation, i.e. with rearranged computation: https://github.com/azliu0/mla-poc/blob/f6a0f35d8e467be36ae038c5a26496e49fde1b9f/pytorch/src/attention.py#L443

see model parameters [here](https://github.com/azliu0/mla-poc/blob/main/config.yml). all benchmarks on a 7B architecture which is roughly deepseek-llm-7b-base (adjusted to support mla).

full results on single gpu `NVIDIA A100 80GB PCIe`:

| model | latency | memory | flops | kv-cache size |
|-------|-------------|------------|-----------|---------------|
| no-cache | 999.01 ms | 26.73 GiB | 1364.08B | n/a |
| kv-cache | 40.44 ms | 27.94 GiB | 33.87B | 1920.0 MiB |
| mla+kv-cache (slow) | 228.44 ms | 27.40 GiB | 1364.08B | 30.0 MiB |
| mla+kv-cache (fast) | 36.24 ms | 26.16 GiB | 35.52B | 30.0 MiB |

## flop analysis

### normal attention
- qkv projection: $b \times s_x \times d_m \times d_h \times n_h \times 3$
- attention scores: $b \times n_h \times s_x \times d_h \times s_t$
- context calculation: $b \times n_h \times s_x \times s_t \times d_h$
- output projection: $b \times s_x \times n_h \times d_h \times d_m$

Total: $b \times n_h \times (4 \times d_h \times d_m \times s_x + 2 \times s_x \times s_t \times d_h)$

### mla
mla with faster implementation:
- downsample: $b \times s_x \times d_m \times d_c$ (negligible without $n_h$ term)
- query projection: $b \times n_h \times s_x \times d_m \times d_c$
- score calculation: $b \times n_h \times s_x \times d_c \times s_t$
- output 1: $b \times n_h \times s_x \times s_t \times d_c$
- output 2: $b \times n_h \times s_x \times d_c \times d_m$

Total: $b \times n_h \times s_x \times (2 \times d_m \times d_c + 2 \times s_t \times d_c)$

With $d_c = 2d_h$, MLA costs: $b \times n_h \times s_x \times (4 \times d_m \times d_h + 4 \times s_t \times d_h)$

Overhead compared to normal attention: $2 \times b \times n_h \times s_x \times s_t \times d_h$

For $(b=1, n_h=64, s_x=5, s_t=2048, d_h=64)$, this adds approximately 2.4B flops across 30 layers, matching empirical measurements (~35.52B vs ~33.87B flops).

Luckily, this is a small overhead relative to the total attention flops, since the $d_m$ term dominates, so we exchange a small amount of flops for memory!
