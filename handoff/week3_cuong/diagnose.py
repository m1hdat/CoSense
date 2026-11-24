"""Diagnose paired NN seed42 dev bundles; affine fit is in-sample only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pandas as pd
from compartment.artifacts import read_bundle


def diagnose(frame):
    rows = []
    for output in ['ModAvg', 'HeadAvg']:
        y = pd.to_numeric(frame['gold_' + output], errors='coerce').to_numpy(float)
        p = pd.to_numeric(frame['pred_' + output], errors='coerce').to_numpy(float)
        valid = np.isfinite(y) & np.isfinite(p)
        y, p = y[valid], p[valid]
        if len(y) < 2:
            raise ValueError(f'At least two finite pairs required: {output}')
        # Centering handles constant predictions without a rank-deficient affine fit.
        centered = p - p.mean()
        slope = float(centered @ (y - y.mean()) / (centered @ centered)) if np.ptp(p) else 0.
        intercept = float(y.mean() - slope * p.mean())
        rmse = float(np.sqrt(np.mean((p - y) ** 2)))
        fitted = float(np.sqrt(np.mean((slope * p + intercept - y) ** 2)))
        rows.append(dict(output=output, n_valid=len(y), n_total=len(frame),
            mean_error=float(np.mean(p - y)),
            pred_gold_sd_ratio=float(p.std() / y.std()) if np.ptp(y) else None,
            rmse=rmse, affine_dev_rmse=fitted, affine_rmse_reduction=rmse-fitted,
            affine_slope=slope, affine_intercept=intercept))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b1', required=True, type=Path)
    parser.add_argument('--b2', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/nn/cuong_diagnostics'))
    args = parser.parse_args()
    frames = []
    provenance = []
    for name, path in [('B1', args.b1), ('B2', args.b2)]:
        meta, _, frame = read_bundle(path)
        if meta['task'] != 'nn' or meta['model'] != name:
            raise ValueError(f'Expected NN {name}: {path}')
        for column, expected in [('task', 'nn'), ('model', name), ('seed', 42), ('split', 'dev')]:
            if set(frame[column]) != {expected}:
                raise ValueError(f'Expected {column}={expected}: {path}')
        frames.append(frame.set_index('context_id').sort_index())
        provenance.append(dict(model=name, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    a, b = frames
    if not a.index.equals(b.index):
        raise ValueError('B1/B2 ID coverage differs')
    for column in ['target', 'context', 'part1', 'part2', 'gold_ModAvg', 'gold_HeadAvg']:
        if not a[column].equals(b[column]):
            raise ValueError(f'B1/B2 mismatch: {column}')
    table = pd.DataFrame([dict(model=name, **row) for name, frame in zip(['B1', 'B2'], frames)
                          for row in diagnose(frame)])
    args.output.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output / 'diagnostics.csv', index=False)
    (args.output / 'provenance.json').write_text(json.dumps(dict(inputs=provenance,
        warning='Affine RMSE fits and evaluates on the same dev rows; diagnostic only, not a model result.',
        decision='Review B2 output reductions against 0.02 before choosing bias or lr. Mixed outputs require a documented decision.'), indent=2))
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
