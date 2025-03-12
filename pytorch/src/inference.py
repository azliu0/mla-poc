import torch
from src.model import TransformerModel
from pathlib import Path
from dataclasses import dataclass
import yaml
import os
import argparse
from typing import Optional

TMP_MODEL_PATH = Path("tmp/model.pth")


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
    )


def do_inference(
    model: TransformerModel, input_ids: torch.Tensor, start_idx: Optional[int] = None
) -> torch.Tensor:
    with torch.no_grad():
        logits = model(input_ids, start_idx)

    return logits


def assert_close(a, b):
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
    mla_logits_no_cache = do_inference(no_cache_model, input_ids)
    mla_logits_cache = do_inference(cache_model, input_ids)

    assert_close(mla_logits_cache, mla_logits_no_cache)

    current_sequence = input_ids.clone()
    current_position = input_ids.shape[1]

    print(f"testing {len(follow_up_batches)} follow-up batches...")

    for i, follow_up_batch in enumerate(follow_up_batches):
        if (i + 1) % 10 == 0:
            print(f"testing batch {i+1}/{len(follow_up_batches)}")

        current_sequence = torch.cat([current_sequence, follow_up_batch], dim=1)
        mla_logits_followup_no_cache = do_inference(no_cache_model, current_sequence)
        mla_logits_followup_no_cache = mla_logits_followup_no_cache[
            :, -follow_up_batch.shape[1] :
        ]

        mla_logits_followup_cache = do_inference(
            cache_model, follow_up_batch, start_idx=current_position
        )

        assert_close(mla_logits_followup_cache, mla_logits_followup_no_cache)
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
    no_mla_logits_no_cache = do_inference(no_cache_model, input_ids)
    no_mla_logits_cache = do_inference(cache_model, input_ids)

    assert_close(no_mla_logits_cache, no_mla_logits_no_cache)

    current_sequence = input_ids.clone()
    current_position = input_ids.shape[1]

    print(f"testing {len(follow_up_batches)} follow-up batches...")

    for i, follow_up_batch in enumerate(follow_up_batches):
        if (i + 1) % 10 == 0:
            print(f"testing batch {i+1}/{len(follow_up_batches)}")

        current_sequence = torch.cat([current_sequence, follow_up_batch], dim=1)
        no_mla_logits_followup_no_cache = do_inference(no_cache_model, current_sequence)
        no_mla_logits_followup_no_cache = no_mla_logits_followup_no_cache[
            :, -follow_up_batch.shape[1] :
        ]

        no_mla_logits_followup_cache = do_inference(
            cache_model, follow_up_batch, start_idx=current_position
        )

        assert_close(no_mla_logits_followup_cache, no_mla_logits_followup_no_cache)
        current_position += follow_up_batch.shape[1]

    print("Non-MLA outputs match with and without cache ✓")

    os.remove(TMP_MODEL_PATH)


def benchmark_no_mla(
    config: ModelConfig, input_ids: torch.Tensor, follow_up_input_ids: torch.Tensor
):
    model = get_model(config, use_cache=False, use_mla=False)
    do_inference(model, input_ids)


def benchmark_mla(
    config: ModelConfig, input_ids: torch.Tensor, follow_up_input_ids: torch.Tensor
):
    model = get_model(config, use_cache=True, use_mla=True)
    do_inference(model, input_ids)


def generate_input(config: ModelConfig) -> tuple[torch.Tensor, list[torch.Tensor]]:
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_length // 2))
    tokens_per_batch = 5
    max_follow_up_batches = 100
    remaining_tokens = config.max_seq_length - input_ids.shape[1]
    num_batches = min(max_follow_up_batches, remaining_tokens // tokens_per_batch)

    follow_up_batches = [
        torch.randint(0, config.vocab_size, (1, tokens_per_batch))
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
        "--no-mla-benchmark",
        action="store_true",
        help="run non-MLA latency and memory benchmark",
    )
    parser.add_argument(
        "--mla-benchmark",
        action="store_true",
        help="run MLA latency and memory benchmark",
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
