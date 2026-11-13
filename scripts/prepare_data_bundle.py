"""Build the audited, frozen Kaggle input bundle from data/train and data/trial."""
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from compartment.bundle import build_bundle  # noqa: E402


if __name__ == "__main__":
    result = build_bundle(
        data_root=REPO_ROOT / "data",
        bundle_dir=REPO_ROOT / "data" / "bundle",
        archive_path=REPO_ROOT / "data" / "cosense-kaggle-data.zip",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
