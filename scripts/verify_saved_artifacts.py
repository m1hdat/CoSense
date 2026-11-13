"""Load saved B1/B2 checkpoints offline and run one CPU forward pass each."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/nn_B1_B2_seed42"),
        help="Directory containing B1/model and B2/model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON; defaults to ROOT/ARTIFACT_SMOKE_TEST.json",
    )
    return parser.parse_args()


def smoke(model_root, variant, torch, nn, AutoConfig, AutoModel, AutoTokenizer):
    required = [
        "best.pt",
        "config.json",
        "run_metadata.json",
        "encoder_config/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ]
    missing = [name for name in required if not (model_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{variant} artifact is incomplete: {missing}")

    metadata = json.loads((model_root / "run_metadata.json").read_text(encoding="utf-8"))
    config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    if (metadata.get("model"), metadata.get("task"), metadata.get("seed")) != (
        variant,
        "nn",
        42,
    ):
        raise ValueError(f"Unexpected {variant} metadata identity")

    class Regressor(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Dropout(0.1), nn.Linear(encoder.config.hidden_size * 2, 2)
            )

        @staticmethod
        def pool(hidden, mask):
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)

        def forward(self, input_ids, attention_mask, part1_mask, part2_mask):
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            pooled = [
                self.pool(hidden, part1_mask),
                self.pool(hidden, part2_mask),
            ]
            return self.head(torch.cat(pooled, dim=-1))

    tokenizer = AutoTokenizer.from_pretrained(
        model_root / "tokenizer", use_fast=True, local_files_only=True
    )
    encoder_config = AutoConfig.from_pretrained(
        model_root / "encoder_config", local_files_only=True
    )
    encoder = AutoModel.from_config(encoder_config)
    model = Regressor(encoder)
    checkpoint = torch.load(
        model_root / "best.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    words = (
        ["flea", "market"]
        if variant == "B1"
        else ["The", "flea", "market", "opened", "today", "."]
    )
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
    )
    word_ids = encoded.word_ids(0)
    first, second = (0, 1) if variant == "B1" else (1, 2)
    part1 = torch.tensor([[int(value == first) for value in word_ids]])
    part2 = torch.tensor([[int(value == second) for value in word_ids]])

    with torch.inference_mode():
        prediction = model(
            encoded["input_ids"], encoded["attention_mask"], part1, part2
        ).squeeze(0)
    values = [float(value) for value in prediction]
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid {variant} prediction: {values}")

    result = {
        "status": "PASS",
        "model": variant,
        "task": "nn",
        "seed": 42,
        "checkpoint_epoch_index": int(checkpoint["epoch"]),
        "prediction": values,
    }
    del model, encoder, checkpoint
    gc.collect()
    return result


def main():
    args = parse_args()
    root = args.root.resolve()
    output = args.output or (root / "ARTIFACT_SMOKE_TEST.json")

    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    checks = [
        smoke(root / variant / "model", variant, torch, nn, AutoConfig, AutoModel, AutoTokenizer)
        for variant in ("B1", "B2")
    ]
    report = {
        "status": "PASS",
        "runtime": {"torch": torch.__version__},
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
