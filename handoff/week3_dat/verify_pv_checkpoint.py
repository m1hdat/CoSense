"""Run the plan's real PV B1 checkpoint check offline on 16 frozen dev rows."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/week3_dat'


def main():
    import numpy as np
    import pandas as pd
    import torch

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    OUT.mkdir(parents=True, exist_ok=True)
    notebook_path = ROOT / 'results/pv/B1_seed42/executed.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    ns = {}
    for index in (6, 7, 8):
        exec(compile(''.join(notebook['cells'][index]['source']),
                     f'{notebook_path.name}:cell{index}', 'exec'), ns)
    archive_path = ROOT / 'results/pv/B1_seed42/complete.zip'
    expected = pd.read_csv(ROOT / 'results/pv/B1_seed42/dev/predictions.tsv',
                           sep='\t', dtype={'context_id': str})
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='pv-check-', dir=OUT) as temp:
        work = Path(temp)
        records, manifest, provenance = ns['load_frozen_task'](
            ROOT / 'data', 'pv', work / 'data', split_seed=42)
        by_id = {r['context_id']: r for r in records}
        ids = expected['context_id'].head(16).tolist()
        assert set(ids).issubset(set(map(str, manifest['partitions']['dev'])))
        assert provenance == json.loads((ROOT / 'results/pv/B1_seed42/data_provenance.json').read_text())
        restored = work / 'checkpoint'
        with zipfile.ZipFile(archive_path) as archive:
            # Resume weights are unnecessary for inference (1.3 GB).
            members = [n for n in archive.namelist() if n == 'best.pt'
                       or n == 'run_metadata.json' or n.startswith(('tokenizer/', 'encoder_config/'))]
            for name in members:
                destination = (restored / name).resolve()
                if not destination.is_relative_to(restored.resolve()):
                    raise ValueError('Unsafe archive member')
                archive.extract(name, restored)
            assert (restored / 'best.pt').is_file()
        predicted, metadata = ns['predict_checkpoint'](restored, [by_id[i] for i in ids])
        assert predicted['context_id'].astype(str).tolist() == ids
        actual = predicted['pred_Avg'].to_numpy(float)
        gold = expected['pred_Avg'].head(16).to_numpy(float)
        assert np.isfinite(actual).all()
        error = float(np.max(np.abs(actual - gold)))
        # CPU vs original T4 inference can differ slightly in float32.
        passed = bool(np.allclose(actual, gold, rtol=1e-5, atol=1e-5))
        pd.DataFrame({'context_id': ids, 'saved_pred_Avg': gold,
                      'reloaded_pred_Avg': actual, 'absolute_difference': np.abs(actual-gold)}).to_csv(
                          OUT / 'pv_b1_inference_16.tsv', sep='\t', index=False)
        result = dict(passed=passed, rows=16, task='pv', model='B1', seed=42,
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      max_absolute_difference=error, atol=1e-5, rtol=1e-5,
                      bitwise_equal=bool(np.array_equal(actual, gold)),
                      elapsed_seconds=round(time.perf_counter()-start, 2),
                      provenance=provenance, torch=torch.__version__,
                      notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                      note='Real saved checkpoint; first 16 stored dev IDs; no training or test evaluation.')
        (OUT / 'pv_b1_inference_verification.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(result, indent=2))
        if not passed:
            raise SystemExit('Checkpoint predictions do not match within tolerance')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
