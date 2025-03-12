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


def load_config(config_path: Path) -> ModelConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[0]
    return ModelConfig(**config["params"])


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
    config: ModelConfig, input_ids: torch.Tensor, follow_up_input_ids: torch.Tensor
):
    os.makedirs("tmp", exist_ok=True)
    no_cache_model = get_model(config, use_cache=False, use_mla=True)
    torch.save(no_cache_model.state_dict(), TMP_MODEL_PATH)

    cache_model = get_model(config, use_cache=True, use_mla=True)
    cache_model.load_state_dict(torch.load(TMP_MODEL_PATH))

    mla_logits_no_cache = do_inference(no_cache_model, input_ids)
    mla_logits_cache = do_inference(cache_model, input_ids)

    assert_close(mla_logits_cache, mla_logits_no_cache)

    # the no-cached model needs to see the entire sequence
    mla_logits_followup_no_cache = do_inference(
        no_cache_model, torch.cat([input_ids, follow_up_input_ids], dim=1)
    )
    mla_logits_followup_no_cache = mla_logits_followup_no_cache[
        :, -follow_up_input_ids.shape[1] :
    ]

    # the cached model only needs to see the new tokens
    mla_logits_followup_cache = do_inference(
        cache_model, follow_up_input_ids, start_idx=input_ids.shape[1]
    )

    assert_close(mla_logits_followup_cache, mla_logits_followup_no_cache)

    print("MLA outputs match with and without cache ✓")

    os.remove(TMP_MODEL_PATH)


def test_no_mla_correctness(
    config: ModelConfig, input_ids: torch.Tensor, follow_up_input_ids: torch.Tensor
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

    total = torch.cat([input_ids, follow_up_input_ids], dim=1)
    no_mla_logits_followup_no_cache = do_inference(no_cache_model, total)
    no_mla_logits_followup_no_cache = no_mla_logits_followup_no_cache[
        :, -follow_up_input_ids.shape[1] :
    ]

    no_mla_logits_followup_cache = do_inference(
        cache_model, follow_up_input_ids, start_idx=input_ids.shape[1]
    )

    assert_close(no_mla_logits_followup_cache, no_mla_logits_followup_no_cache)

    print("Non-MLA outputs match with and without cache ✓")

    os.remove(TMP_MODEL_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-mla-correctness",
        action="store_true",
        help="Run non-MLA correctness test",
    )
    parser.add_argument(
        "--mla-correctness",
        action="store_true",
        help="Run MLA correctness test",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent.parent / "config.yml"
    config = load_config(config_path)
    print(config)

    input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_length // 2))
    follow_up_input_ids = torch.randint(
        0, config.vocab_size, (1, config.max_seq_length // 2)
    )

    if args.no_mla_correctness:
        test_no_mla_correctness(config, input_ids, follow_up_input_ids)
    if args.mla_correctness:
        test_mla_correctness(config, input_ids, follow_up_input_ids)
