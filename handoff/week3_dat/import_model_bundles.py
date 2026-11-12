"""Validate and retain inference bundles; replay 16 dev rows per real checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/week3_dat'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    import gc
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    ns = {}
    notebook = json.loads((ROOT / 'notebooks/NN_B1_kaggle.ipynb').read_text(encoding='utf-8'))
    for index in (6, 7, 8):
        exec(compile(''.join(notebook['cells'][index]['source']), f'local_helper_{index}', 'exec'), ns)
    checks = []
    for task in ('nn', 'pv'):
        source = Path.home() / f'Downloads/{task.upper()}_all_models_results.zip'
        destination = ROOT / f'results/{task}/models_seeds43_44.zip'
        expected_hash = file_hash(source)
        if destination.exists():
            assert file_hash(destination) == expected_hash, 'Existing model bundle differs; preserve both'
        else:
            shutil.copyfile(source, destination)
            assert file_hash(destination) == expected_hash
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
            inventory = json.loads(archive.read('DOWNLOAD_CONTENTS.json'))
            assert inventory['task'] == task and not inventory['includes_resume']
            assert len(set(archive.namelist())) == len(archive.namelist())
            assert {f['name'] for f in inventory['files']} == set(archive.namelist()) - {'DOWNLOAD_CONTENTS.json'}
            for entry in inventory['files']:
                assert archive.getinfo(entry['name']).file_size == entry['bytes']
            for seed in (43, 44):
                for model in ('B1', 'B2'):
                    prefix = f'{task.upper()}_{model}_seed{seed}_full/'
                    run = ROOT / f'results/{task}/{model}_seed{seed}'
                    for name in ('dev/predictions.tsv', 'dev/metrics.json', 'dev/run_metadata.json'):
                        assert archive.read(prefix+name) == (run/name).read_bytes()
                    metadata = json.loads(archive.read(prefix+'run_metadata.json'))
                    reference = json.loads((run/'run_metadata.json').read_text())
                    assert metadata == {k: v for k, v in reference.items() if k != 'contract_version'}
                    for name in ('data_provenance.json', 'split_manifest.json'):
                        assert json.loads(archive.read(prefix+name)) == json.loads((run/name).read_text())
                    with tempfile.TemporaryDirectory(prefix='model-replay-', dir=OUT) as temp:
                        folder = Path(temp)
                        members = [i for i in archive.namelist() if i.startswith(prefix) and
                                   (i[len(prefix):] in ('best.pt', 'run_metadata.json') or
                                    i[len(prefix):].startswith(('tokenizer/', 'encoder_config/')))]
                        for member in members:
                            local = folder / member[len(prefix):]
                            assert local.resolve().is_relative_to(folder.resolve())
                            local.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(member) as src, local.open('wb') as dst:
                                shutil.copyfileobj(src, dst)
                        best_hash = file_hash(folder/'best.pt')
                        records, manifest, provenance = ns['load_frozen_task'](
                            ROOT/'data', task, folder/'data', split_seed=42)
                        by_id = {r['context_id']: r for r in records}
                        frame = pd.read_csv(run/'dev/predictions.tsv', sep='\t', dtype={'context_id': str}).head(16)
                        ids = frame.context_id.tolist()
                        actual, _ = ns['predict_checkpoint'](folder, [by_id[i] for i in ids])
                        assert actual.context_id.astype(str).tolist() == ids
                        columns = ['pred_'+o for o in ns['SCHEMAS'][task]['labels']]
                        a = actual[columns].to_numpy(float); b = frame[columns].to_numpy(float)
                        error = float(np.max(np.abs(a-b)))
                        passed = bool(np.allclose(a, b, rtol=1e-5, atol=1e-5))
                        assert passed, (task, model, seed, error)
                    location = dict(archive=str(destination.relative_to(ROOT)), prefix=prefix,
                                    archive_sha256=expected_hash, best_pt_sha256=best_hash,
                                    kind='inference', includes_resume=False,
                                    checkpoint_crc_ok=True, dev_predictions_match=True,
                                    inference_16_rows_passed=passed, max_absolute_difference=error)
                    (run/'MODEL_ARCHIVE.json').write_text(json.dumps(location, indent=2)+'\n', encoding='utf-8')
                    checks.append(dict(task=task, model=model, seed=seed, **location))
                    (OUT/'model_bundle_verification.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
                    print(f'PASS {task} {model} seed{seed}: CRC, metadata, predictions, real inference 16 rows; max diff {error:.3g}', flush=True)
                    gc.collect()


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
