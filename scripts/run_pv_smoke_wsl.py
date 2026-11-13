"""Run a small CPU PV training smoke test from the standalone Kaggle notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_notebook_code(model: str) -> dict:
    path = REPO_ROOT / f"notebooks/{model}_kaggle.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "cosense_pv_smoke"}
    for index in (6, 7, 8):
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise ValueError(f"Expected code cell {index} in {path}")
        exec("".join(cell["source"]), namespace)
    return namespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["B1", "B2"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path; defaults under results/pv_smoke_wsl.",
    )
    args = parser.parse_args()
    output = args.output or REPO_ROOT / f"results/pv_smoke_wsl/{args.model}_seed42.json"

    namespace = load_notebook_code(args.model)
    records = namespace["load_records"](REPO_ROOT / "data/bundle/en-pv-train.tsv", "pv")
    manifest = json.loads(
        (REPO_ROOT / "data/bundle/split_manifest_pv.json").read_text(encoding="utf-8")
    )
    config = {
        "backbone": "bert-base-uncased",
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "seed": 42,
        "lr": 2e-5,
        "batch_size": 8,
        "accumulation": 2,
        "epochs": 1,
        "patience": 2,
        "max_length": 128,
        "layer": -1,
        "smoke": True,
        "smoke_rows": 16,
        "allow_cpu": True,
        "fallback": "expression",
        "log_every_batches": 20,
    }
    with tempfile.TemporaryDirectory(prefix=f"cosense_pv_{args.model.lower()}_") as temporary:
        run_root = Path(temporary)
        _, _, metadata = namespace["train"](
            records,
            manifest,
            config,
            args.model,
            run_root,
            evaluate_test=False,
        )
        metrics = json.loads((run_root / "dev/metrics.json").read_text(encoding="utf-8"))
        summary = {
            "status": "PASS",
            "task": "pv",
            "model": args.model,
            "seed": 42,
            "smoke": True,
            "train_rows": 16,
            "dev_rows": 16,
            "best_epoch_index": metadata["best_epoch"],
            "preprocessing": metadata["preprocessing"],
            "metrics": metrics,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
