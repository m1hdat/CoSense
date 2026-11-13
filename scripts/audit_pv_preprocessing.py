"""Audit PV alignment and B2 tokenization using the exact Kaggle notebook code."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def notebook_namespace(notebook_path: Path) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "cosense_pv_audit"}
    for index in (6, 8):
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Expected code cell {index} in {notebook_path}")
        exec("".join(cell["source"]), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "results/nn_B1_B2_seed42/B2/model/tokenizer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/pv_preprocessing_audit_seed42.json",
    )
    args = parser.parse_args()

    namespace = notebook_namespace(REPO_ROOT / "notebooks/B2_kaggle.ipynb")
    records = namespace["load_records"](REPO_ROOT / "data/bundle/en-pv-train.tsv", "pv")
    manifest = json.loads(
        (REPO_ROOT / "data/bundle/split_manifest_pv.json").read_text(encoding="utf-8")
    )
    namespace["validate_splits"](records, manifest)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=True, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required")

    result = {
        "status": "PASS",
        "task": "pv",
        "model": "B2",
        "split_seed": manifest["seed"],
        "max_length": 128,
        "splits": {},
    }
    for split, ids in manifest["partitions"].items():
        id_set = set(ids)
        subset = [record for record in records if record["context_id"] in id_set]
        dataset = namespace["EncodedDataset"](
            subset,
            tokenizer,
            {"max_length": 128, "fallback": "expression"},
            "B2",
        )
        fallback_reasons = Counter(
            log["fallback_reason"] for log in dataset.logs if log["fallback_reason"]
        )
        result["splits"][split] = {
            "rows": len(subset),
            "alignment": dict(Counter(r["alignment_status"] for r in subset)),
            "fallback": sum(log["fallback_reason"] is not None for log in dataset.logs),
            "fallback_reasons": dict(fallback_reasons),
            "truncated": sum(log["truncated"] for log in dataset.logs),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
