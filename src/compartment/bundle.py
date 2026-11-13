"""Audit official data, freeze target-disjoint splits, and build the Kaggle input ZIP."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import zipfile

from .data import SCHEMAS, audit, digest, dump, load_records
from .splits import make_splits, validate_splits


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    language: str
    task: str
    split: str
    rows: int
    targets: int


OFFICIAL_SPECS = (
    DatasetSpec("train/en-nn-train.tsv", "en", "nn", "train", 3480, 444),
    DatasetSpec("train/en-pv-train.tsv", "en", "pv", "train", 1557, 158),
    DatasetSpec("train/de-nn-train.tsv", "de", "nn", "train", 3298, 461),
    DatasetSpec("train/de-pv-train.tsv", "de", "pv", "train", 1490, 158),
    DatasetSpec("trial/en-nn-trial.tsv", "en", "nn", "trial", 10, 2),
    DatasetSpec("trial/en-pv-trial.tsv", "en", "pv", "trial", 10, 2),
)

BUNDLE_MEMBERS = (
    "README.md",
    "dataset_info.json",
    "en-nn-train.tsv",
    "en-pv-train.tsv",
    "split_manifest_nn.json",
    "split_manifest_pv.json",
)


def audit_dataset(path, spec: DatasetSpec):
    path = Path(path)
    records = load_records(path, spec.task)
    summary = audit(records)
    if summary["rows"] != spec.rows or summary["targets"] != spec.targets:
        raise ValueError(
            f"Unexpected size for {path}: got {summary['rows']} rows/"
            f"{summary['targets']} targets, expected {spec.rows}/{spec.targets}"
        )
    return records, {
        "path": spec.path.replace("\\", "/"),
        "language": spec.language,
        "task": spec.task,
        "split": spec.split,
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "rows": summary["rows"],
        "targets": summary["targets"],
        "columns": [
            "ContextID",
            SCHEMAS[spec.task]["target"],
            *SCHEMAS[spec.task]["parts"],
            *SCHEMAS[spec.task]["labels"],
            *SCHEMAS[spec.task]["std"],
            "Context",
        ],
        "alignment": summary["alignment"],
        "labels": summary["labels"],
        "annotation_std": summary["annotation_std"],
    }

def audit_inputs(data_root, specs=OFFICIAL_SPECS):
    data_root = Path(data_root)
    records = {}
    files = {}
    for spec in specs:
        key = (spec.language, spec.task, spec.split)
        if key in records:
            raise ValueError(f"Duplicate dataset specification: {key}")
        item_records, item_info = audit_dataset(data_root / spec.path, spec)
        records[key] = item_records
        files[spec.path.replace("\\", "/")] = item_info

    separation = {}
    for language, task in sorted({(s.language, s.task) for s in specs}):
        train = records.get((language, task, "train"))
        trial = records.get((language, task, "trial"))
        if train is None or trial is None:
            continue
        train_ids = {r["context_id"] for r in train}
        trial_ids = {r["context_id"] for r in trial}
        train_targets = {r["target"] for r in train}
        trial_targets = {r["target"] for r in trial}
        id_overlap = sorted(train_ids & trial_ids)
        target_overlap = sorted(train_targets & trial_targets)
        if id_overlap or target_overlap:
            raise ValueError(
                f"{language}-{task} train/trial overlap: "
                f"IDs={id_overlap[:5]}, targets={target_overlap[:5]}"
            )
        separation[f"{language}-{task}"] = {
            "id_overlap": 0,
            "target_overlap": 0,
            "trial_excluded_from_training": True,
        }
    return records, files, separation


def _write_deterministic_zip(path, folder):
    path = Path(path)
    folder = Path(folder)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in BUNDLE_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (folder / name).read_bytes())


def build_bundle(data_root="data", bundle_dir="data/bundle", archive_path="data/cosense-kaggle-data.zip",
                 split_seed=42, specs=OFFICIAL_SPECS):
    data_root = Path(data_root)
    bundle_dir = Path(bundle_dir)
    archive_path = Path(archive_path)
    records, audited_files, separation = audit_inputs(data_root, specs)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    files_sha256 = {}
    manifests = {}
    split_summary = {}
    for task in ("nn", "pv"):
        source = data_root / "train" / f"en-{task}-train.tsv"
        destination = bundle_dir / source.name
        shutil.copyfile(source, destination)
        files_sha256[source.name] = digest(destination)

        task_records = records[("en", task, "train")]
        manifest = make_splits(task_records, seed=split_seed)
        manifest.update(source_file=source.name, source_sha256=digest(source))
        validate_splits(task_records, manifest)
        manifest_name = f"split_manifest_{task}.json"
        dump(bundle_dir / manifest_name, manifest)
        files_sha256[manifest_name] = digest(bundle_dir / manifest_name)
        manifests[task] = manifest
        by_id = {r["context_id"]: r for r in task_records}
        split_summary[task] = {
            name: {
                "rows": len(ids),
                "targets": len({by_id[i]["target"] for i in ids}),
            }
            for name, ids in manifest["partitions"].items()
        }

    info = {
        "bundle_version": 1,
        "created_from": "SemEval-2027 Task 5 ComPartMent CodaBench competition 17971",
        "official_reference": "https://www.codabench.org/competitions/17971/",
        "verified_date": "2026-11-12",
        "scope": "English official training data only; trial is audited but excluded",
        "split_seed": split_seed,
        "split_policy": manifests["nn"]["policy"],
        "files_sha256": files_sha256,
        "datasets": audited_files,
        "train_trial_separation": separation,
        "internal_splits": split_summary,
    }
    dump(bundle_dir / "dataset_info.json", info)
    readme = (
        "# CoSense frozen Kaggle data\n\n"
        "This bundle contains the official English training TSV files and target-disjoint "
        "internal train/dev/test manifests generated with split seed 42. Trial data is not "
        "included and must not be used for training or model selection. Raw TSV bytes are "
        "unchanged; verify them and the manifests with `dataset_info.json`.\n"
    )
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")
    _write_deterministic_zip(archive_path, bundle_dir)

    with zipfile.ZipFile(archive_path) as archive:
        if tuple(sorted(archive.namelist())) != tuple(sorted(BUNDLE_MEMBERS)):
            raise ValueError("Unexpected Kaggle ZIP members")
    return {
        "bundle_dir": str(bundle_dir),
        "archive": str(archive_path),
        "archive_sha256": digest(archive_path),
        "internal_splits": split_summary,
        "files": Counter(s.split("/")[0] for s in audited_files),
    }
