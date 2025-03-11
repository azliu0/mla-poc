import torch
from src.model import TransformerModel
from pathlib import Path
from dataclasses import dataclass
import yaml


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    d_head: int
    num_heads: int
    num_layers: int
    max_seq_length: int
    d_kv_latent: int


def load_config(config_path: Path) -> ModelConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[0]
    return ModelConfig(**config["params"])


def do_inference(config: ModelConfig, use_mla: bool = True):
    model = TransformerModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_head=config.d_head,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_length=config.max_seq_length,
        d_kv_latent=config.d_kv_latent if use_mla else None,
    )

    input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_length))

    with torch.no_grad():
        logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")

    predictions = torch.argmax(logits, dim=-1)
    print(f"Predictions shape: {predictions.shape}")

    return model, logits


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent.parent / "config.yml"
    config = load_config(config_path)
    print(config)

    print("running inference with MLA:")
    mla_model, mla_logits = do_inference(config, use_mla=True)

    print("running inference without MLA:")
    no_mla_model, no_mla_logits = do_inference(config, use_mla=False)
