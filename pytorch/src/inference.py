import torch
from src.model import TransformerModel
from pathlib import Path
from dataclasses import dataclass
import yaml
import os
import argparse
from typing import Optional
import modal
from fvcore.nn import FlopCountAnalysis  # type: ignore

TMP_MODEL_PATH = Path("tmp/model.pth")
REQUIREMENTS_PATH = Path(__file__).parent.parent / "requirements.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install_from_requirements(str(REQUIREMENTS_PATH))
)
app = modal.App(name="mla-poc-pytorch", image=image)


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    d_head: int
    num_heads: int
    num_layers: int
    max_seq_length: int
    d_kv_latent: int
    max_batch_size: int


@dataclass
class Summary:
    name: str
    initial_latency_ms: float
    initial_memory_gib: float
    avg_latency_ms: float
    avg_memory_gib: float
    avg_flops: float


@dataclass
class InferenceOutput:
    logits: Optional[torch.Tensor]
    flops: int


def load_config(config_path: Path, small: bool) -> ModelConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[0]
    if small:
        return ModelConfig(**config["small"])
    else:
        return ModelConfig(**config["large"])


def get_model(config: ModelConfig, use_cache: bool, use_mla: bool) -> TransformerModel:
    return TransformerModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_head=config.d_head,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_length=config.max_seq_length,
        d_kv_latent=config.d_kv_latent if use_mla else None,
        max_batch_size=config.max_batch_size,
        use_cache=use_cache,
        device=DEVICE,
    ).to(DEVICE)


def do_inference(
    model: TransformerModel,
    input_ids: torch.Tensor,
    start_idx: Optional[int],
    count_flops: bool = False,
) -> InferenceOutput:
    with torch.no_grad():
        if count_flops:
            logits = None
            flops = FlopCountAnalysis(model, (input_ids, start_idx)).total()
        else:
            logits = model(input_ids, start_idx)
            flops = 0

    return InferenceOutput(logits=logits, flops=flops)


def assert_close(a: torch.Tensor, b: torch.Tensor):
    assert torch.allclose(a, b, atol=1e-5), "logits differ!"


def test_mla_correctness(
    config: ModelConfig, input_ids: torch.Tensor, follow_up_batches: list[torch.Tensor]
):
    os.makedirs("tmp", exist_ok=True)
    no_cache_model = get_model(config, use_cache=False, use_mla=True)
    torch.save(no_cache_model.state_dict(), TMP_MODEL_PATH)

    cache_model = get_model(config, use_cache=True, use_mla=True)
    cache_model.load_state_dict(torch.load(TMP_MODEL_PATH))

    print("running correctness test for model with MLA:")
    mla_logits_no_cache = do_inference(
        no_cache_model, input_ids, start_idx=None, count_flops=False
    )
    mla_logits_cache = do_inference(
        cache_model, input_ids, start_idx=None, count_flops=False
    )
    assert (
        mla_logits_cache.logits is not None and mla_logits_no_cache.logits is not None
    )
    assert_close(mla_logits_cache.logits, mla_logits_no_cache.logits)

    current_sequence = input_ids.clone()
    current_position = input_ids.shape[1]

    print(f"testing {len(follow_up_batches)} follow-up batches...")

    for i, follow_up_batch in enumerate(follow_up_batches):
        if (i + 1) % 10 == 0:
            print(f"testing batch {i+1}/{len(follow_up_batches)}")

        current_sequence = torch.cat([current_sequence, follow_up_batch], dim=1)
        mla_logits_followup_no_cache = do_inference(
            no_cache_model, current_sequence, start_idx=None, count_flops=False
        )
        assert mla_logits_followup_no_cache.logits is not None
        mla_logits_followup_no_cache.logits = mla_logits_followup_no_cache.logits[
            :, -follow_up_batch.shape[1] :
        ]

        mla_logits_followup_cache = do_inference(
            cache_model, follow_up_batch, start_idx=current_position, count_flops=False
        )
        assert mla_logits_followup_cache.logits is not None
        assert_close(
            mla_logits_followup_cache.logits, mla_logits_followup_no_cache.logits
        )
        current_position += follow_up_batch.shape[1]

    print("MLA outputs match with and without cache ✓")

    os.remove(TMP_MODEL_PATH)


def test_no_mla_correctness(
    config: ModelConfig, input_ids: torch.Tensor, follow_up_batches: list[torch.Tensor]
):
    os.makedirs("tmp", exist_ok=True)
    no_cache_model = get_model(config, use_cache=False, use_mla=False)
    torch.save(no_cache_model.state_dict(), TMP_MODEL_PATH)

    cache_model = get_model(config, use_cache=True, use_mla=False)
    cache_model.load_state_dict(torch.load(TMP_MODEL_PATH))

    print("running correctness test for model without MLA:")
    no_mla_logits_no_cache = do_inference(
        no_cache_model, input_ids, start_idx=None, count_flops=False
    )
    no_mla_logits_cache = do_inference(
        cache_model, input_ids, start_idx=None, count_flops=False
    )
    assert (
        no_mla_logits_cache.logits is not None
        and no_mla_logits_no_cache.logits is not None
    )
    assert_close(no_mla_logits_cache.logits, no_mla_logits_no_cache.logits)

    current_sequence = input_ids.clone()
    current_position = input_ids.shape[1]

    print(f"testing {len(follow_up_batches)} follow-up batches...")

    for i, follow_up_batch in enumerate(follow_up_batches):
        if (i + 1) % 10 == 0:
            print(f"testing batch {i+1}/{len(follow_up_batches)}")

        current_sequence = torch.cat([current_sequence, follow_up_batch], dim=1)
        no_mla_logits_followup_no_cache = do_inference(
            no_cache_model, current_sequence, start_idx=None, count_flops=False
        )
        assert no_mla_logits_followup_no_cache.logits is not None
        no_mla_logits_followup_no_cache.logits = no_mla_logits_followup_no_cache.logits[
            :, -follow_up_batch.shape[1] :
        ]

        no_mla_logits_followup_cache = do_inference(
            cache_model, follow_up_batch, start_idx=current_position, count_flops=False
        )
        assert no_mla_logits_followup_cache.logits is not None
        assert_close(
            no_mla_logits_followup_cache.logits, no_mla_logits_followup_no_cache.logits
        )
        current_position += follow_up_batch.shape[1]

    print("Non-MLA outputs match with and without cache ✓")

    os.remove(TMP_MODEL_PATH)


def run_benchmark(
    name: str,
    config: ModelConfig,
    input_ids: torch.Tensor,
    follow_up_batches: list[torch.Tensor],
    use_cache: bool,
    use_mla: bool,
    count_flops: bool,
    limit_batches=None,
) -> Summary:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = get_model(config, use_cache=use_cache, use_mla=use_mla)
    print("num params:", sum(p.numel() for p in model.parameters()))

    batches_to_use = follow_up_batches
    if limit_batches is not None:
        batches_to_use = follow_up_batches[:limit_batches]

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    start_time.record(stream=torch.cuda.current_stream())
    _ = do_inference(model, input_ids, start_idx=None, count_flops=count_flops)
    end_time.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    initial_latency_ms = start_time.elapsed_time(end_time)
    initial_memory_gib = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

    print(f"{name} setup:")
    print(f"  - latency: {initial_latency_ms:.2f} ms")
    print(f"  - memory: {initial_memory_gib:.2f} GiB")

    current_position = input_ids.shape[1]
    follow_up_latencies = []
    follow_up_memories = []
    follow_up_flops = []

    current_sequence = input_ids.clone() if not model.use_cache else None

    for i, follow_up_batch in enumerate(batches_to_use):
        follow_up_batch = follow_up_batch.to(DEVICE)

        torch.cuda.reset_peak_memory_stats()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record(stream=torch.cuda.current_stream())

        if model.use_cache:
            flops = do_inference(
                model,
                follow_up_batch,
                start_idx=current_position,
                count_flops=count_flops,
            ).flops
        else:
            assert current_sequence is not None
            current_sequence = torch.cat([current_sequence, follow_up_batch], dim=1)
            flops = do_inference(
                model, current_sequence, start_idx=None, count_flops=count_flops
            ).flops

        follow_up_flops.append(flops)

        end_time.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()

        batch_latency_ms = start_time.elapsed_time(end_time)
        batch_memory_gib = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)

        follow_up_latencies.append(batch_latency_ms)
        follow_up_memories.append(batch_memory_gib)

        current_position += follow_up_batch.shape[1]

        if (i + 1) % 10 == 0 or i == 0 or i == len(batches_to_use) - 1:
            print(f"{name} batch {i+1}/{len(batches_to_use)}:")
            print(f"  - latency: {batch_latency_ms:.2f} ms")
            print(f"  - memory: {batch_memory_gib:.2f} GiB")

    avg_latency = sum(follow_up_latencies) / len(follow_up_latencies)
    avg_memory = sum(follow_up_memories) / len(follow_up_memories)
    avg_flops = sum(follow_up_flops) / len(follow_up_flops)

    summary = Summary(
        name=name,
        initial_latency_ms=initial_latency_ms,
        initial_memory_gib=initial_memory_gib,
        avg_latency_ms=avg_latency,
        avg_memory_gib=avg_memory,
        avg_flops=avg_flops,
    )

    del model
    del current_sequence
    torch.cuda.empty_cache()

    return summary


@app.function(gpu="A100-80GB")
def benchmark_models(
    config: ModelConfig,
    input_ids: torch.Tensor,
    follow_up_batches: list[torch.Tensor],
    count_flops: bool,
):
    print(f"running benchmarks on {torch.cuda.get_device_name()}")
    print(
        f"model config: d_model={config.d_model}, layers={config.num_layers}, heads={config.num_heads}"
    )
    print(
        f"sequence length: initial={input_ids.shape[1]}, follow-up batches={len(follow_up_batches)}"
    )
    print("-" * 80)

    # input_ids lose their device during modal's pickling
    input_ids = input_ids.to(DEVICE)

    summaries: list[Summary] = []

    summaries.append(
        run_benchmark(
            "no-cache",
            config,
            input_ids,
            follow_up_batches,
            use_cache=False,
            use_mla=False,
            count_flops=count_flops,
            limit_batches=20,
        )
    )

    summaries.append(
        run_benchmark(
            "mla+kv-cache",
            config,
            input_ids,
            follow_up_batches,
            use_cache=True,
            use_mla=True,
            count_flops=count_flops,
        )
    )

    summaries.append(
        run_benchmark(
            "kv-cache",
            config,
            input_ids,
            follow_up_batches,
            use_cache=True,
            use_mla=False,
            count_flops=count_flops,
        )
    )

    print("\n\n")
    print(
        f"{'model':<15} {'initial latency':<20} {'initial memory':<20} {'avg latency':<20} {'avg memory':<20} {'avg flops':<20}"
    )
    print("-" * 135)

    for summary in summaries:
        print(
            f"{summary.name:<15} {summary.initial_latency_ms:.2f} ms{'':<10} {summary.initial_memory_gib:.2f} GiB{'':<10} {summary.avg_latency_ms:.2f} ms{'':<10} {summary.avg_memory_gib:.2f} GiB{'':<10} {summary.avg_flops / 1e9:.2f}B{'':<10}"
        )

    kv_cache_size = (
        2
        * config.max_batch_size
        * config.max_seq_length
        * config.num_heads
        * config.d_head
        * 4  # bytes per param
        / 1024
        / 1024
    )
    mla_kv_cache_size = (
        config.max_batch_size
        * config.max_seq_length
        * config.d_kv_latent
        * 4  # bytes per param
        / 1024
        / 1024
    )
    print("\n\n")
    print(
        "mla kv-cache size",
        mla_kv_cache_size,
        "MiB",
    )
    print(
        "kv-cache size",
        kv_cache_size,
        "MiB",
    )
    print(f"{'% reduction':<15} {(1 - mla_kv_cache_size / kv_cache_size) * 100:.2f}%")


def generate_input(config: ModelConfig) -> tuple[torch.Tensor, list[torch.Tensor]]:
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_length // 2)).to(
        DEVICE
    )
    tokens_per_batch = 5
    max_follow_up_batches = 100
    remaining_tokens = config.max_seq_length - input_ids.shape[1]
    num_batches = min(max_follow_up_batches, remaining_tokens // tokens_per_batch)

    follow_up_batches = [
        torch.randint(0, config.vocab_size, (1, tokens_per_batch)).to(DEVICE)
        for _ in range(num_batches)
    ]

    return input_ids, follow_up_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-mla-correctness",
        action="store_true",
        help="run non-MLA correctness test",
    )
    parser.add_argument(
        "--mla-correctness",
        action="store_true",
        help="run MLA correctness test",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run comprehensive benchmarks comparing no-cache, KV-cache, and MLA",
    )
    parser.add_argument(
        "--count-flops",
        action="store_true",
        help="count FLOPs",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent.parent / "config.yml"
    small_config = load_config(config_path, small=True)
    large_config = load_config(config_path, small=False)

    input_ids, follow_up_batches = generate_input(small_config)

    if args.no_mla_correctness:
        test_no_mla_correctness(small_config, input_ids, follow_up_batches)
    if args.mla_correctness:
        test_mla_correctness(small_config, input_ids, follow_up_batches)
    if args.benchmark:
        with modal.enable_output():
            with app.run():
                benchmark_models.remote(
                    large_config,
                    input_ids,
                    follow_up_batches,
                    count_flops=args.count_flops,
                )

# rn the mla flops are inflated:
# latency-mem-flops:
# mla+kv-cache    228.44 ms           27.40 GiB           1364.08B
# kv-cache        41.73 ms           27.94 GiB           33.87B
# TODO: figure out why

# flop calculation:
# remember d_c=2d_h

# normal:
# - qkv: bs_xd_md_hn_h * 3
# - scores: bn_hs_xd_hs_t
# - context: bn_hs_xs_td_h
# - out: bs_xn_hd_hd_m

# total:
# - bn_h
# 	- 4\*d_hd_ms_x
# 		- MLS
# 	- 2\*s_xs_td_h
# 		- MLS

# mla:
# - downsample: bs_xd_md_c
# 	- ignore for now
# - q_part: bn_hs_xd_md_c
# - v_part: bn_hd_md_cs_t <- this is really bad, i think
# - scores: bn_hs_xd_cs_t
# - out: bn_hs_xs_td_m

# total:
# - bn_h
# 	- s_xs_t
# 		- d_m+d_c=3d_h
# 		- MLS
# 	- d_cs_td_m
# 		- this is rly ugly bc of s_td_m
# 		- MLL
# 	- s_xd_md_c
# 		- same as 2\*d_hd_ms_x
# 		- MLS
