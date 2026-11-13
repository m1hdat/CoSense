"""Synthetic tests for the reproducible data audit and Kaggle bundle builder."""
import csv
import json
from pathlib import Path
import zipfile

from compartment.bundle import BUNDLE_MEMBERS, DatasetSpec, build_bundle
from compartment.data import load_records
from compartment.splits import validate_splits


def _write_tsv(path, task, targets, contexts):
    path.parent.mkdir(parents=True, exist_ok=True)
    if task == "nn":
        header = ["ContextID", "Compound", "Mod", "Head", "ModAvg", "ModStd", "HeadAvg", "HeadStd", "Context"]
    else:
        header = ["ContextID", "ParticleVerb", "Base", "Particle", "Avg", "Std", "Context"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for target in range(targets):
            first = f"{path.parent.name}word{target}"
            second = "book" if task == "nn" else "off"
            for context in range(contexts):
                cid = f"{task}-{path.stem}-{target}-{context}"
                text = f"Example {context} contains {first} {second} today."
                if task == "nn":
                    writer.writerow([cid, f"{first} {second}", first, second, 2.0, 0.5, 3.0, 0.4, text])
                else:
                    writer.writerow([cid, f"{first} {second}", first, second, 2.0, 0.5, text])


def test_bundle_is_frozen_target_disjoint_and_excludes_trial(tmp_path):
    data = tmp_path / "data"
    specs = []
    for task in ("nn", "pv"):
        train = data / "train" / f"en-{task}-train.tsv"
        trial = data / "trial" / f"en-{task}-trial.tsv"
        _write_tsv(train, task, targets=7, contexts=2)
        _write_tsv(trial, task, targets=2, contexts=1)
        specs.extend([
            DatasetSpec(f"train/en-{task}-train.tsv", "en", task, "train", 14, 7),
            DatasetSpec(f"trial/en-{task}-trial.tsv", "en", task, "trial", 2, 2),
        ])

    first = build_bundle(data, tmp_path / "bundle1", tmp_path / "one.zip", specs=tuple(specs))
    second = build_bundle(data, tmp_path / "bundle2", tmp_path / "two.zip", specs=tuple(specs))
    assert first["archive_sha256"] == second["archive_sha256"]

    with zipfile.ZipFile(tmp_path / "one.zip") as archive:
        assert set(archive.namelist()) == set(BUNDLE_MEMBERS)
        assert not any("trial" in name for name in archive.namelist())
        assert all("/" not in name for name in archive.namelist())

    info = json.loads((tmp_path / "bundle1/dataset_info.json").read_text(encoding="utf-8"))
    assert info["split_seed"] == 42
    assert info["train_trial_separation"]["en-nn"]["target_overlap"] == 0
    for task in ("nn", "pv"):
        records = load_records(data / "train" / f"en-{task}-train.tsv", task)
        manifest = json.loads((tmp_path / f"bundle1/split_manifest_{task}.json").read_text(encoding="utf-8"))
        assert manifest["source_file"] == f"en-{task}-train.tsv"
        assert validate_splits(records, manifest)
