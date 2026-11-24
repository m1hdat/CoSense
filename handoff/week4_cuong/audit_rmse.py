"""Audit four NN seed42 dev bundles without loading models or using dev calibration as a result."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import zipfile


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            result[index] = (start + end + 1) / 2
        start = end
    return result


def error_decomposition(gold, pred):
    if not gold or len(gold) != len(pred) or not all(math.isfinite(v) for v in gold + pred):
        raise ValueError('Expected nonempty, finite paired arrays')
    errors = [p-y for y, p in zip(gold, pred)]
    bias = statistics.mean(errors)
    mse = statistics.mean(e*e for e in errors)
    centered_mse = statistics.mean((e-bias)**2 for e in errors)
    if not math.isclose(mse, bias*bias + centered_mse, abs_tol=1e-12):
        raise ValueError('MSE decomposition failed')
    return dict(rmse=math.sqrt(mse), mae=statistics.mean(abs(e) for e in errors),
                mse=mse, mean_error=bias, squared_mean_error=bias*bias,
                centered_mse=centered_mse, centered_rmse=math.sqrt(centered_mse),
                pred_gold_sd_ratio=statistics.pstdev(pred)/statistics.pstdev(gold),
                outside_0_5=sum(not 0 <= v <= 5 for v in pred),
                spearman=statistics.correlation(ranks(gold), ranks(pred)))


def audit(baseline_dir, bias_dir):
    rows, epochs, sources = [], [], []
    reference, reference_manifest = None, None
    for variant, directory in [('baseline', baseline_dir), ('bias', bias_dir)]:
        for model in ['B1', 'B2']:
            path = directory / f'NN_{model}_seed42_full_results.zip'
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f'CRC failed: {path}')
                meta = json.loads(archive.read('run_metadata.json'))
                saved = json.loads(archive.read('metrics.json'))
                manifest = json.loads(archive.read('split_manifest.json'))
                frame = list(csv.DictReader(io.StringIO(archive.read('predictions.tsv').decode()), delimiter='\t'))
                history = list(csv.DictReader(io.StringIO(archive.read('training_history.csv').decode())))
            if (meta['task'], meta['model'], meta['seed'], meta['config']['smoke']) != ('nn', model, 42, False):
                raise ValueError('Unexpected metadata')
            identities = sorted((r['context_id'], r['target'], r['context'], r['gold_ModAvg'], r['gold_HeadAvg']) for r in frame)
            if reference is None:
                reference, reference_manifest = identities, manifest
            if identities != reference or manifest != reference_manifest:
                raise ValueError('ID/gold/context/split mismatch')
            if set(meta['actual_split_ids']) != {'train', 'dev'} or any(ids != manifest['partitions'][s] for s, ids in meta['actual_split_ids'].items()):
                raise ValueError('Actual split IDs mismatch')
            for output in ['ModAvg', 'HeadAvg']:
                gold = [float(r['gold_'+output]) for r in frame]
                pred = [float(r['pred_'+output]) for r in frame]
                metrics = error_decomposition(gold, pred)
                for key in ['rmse', 'mae', 'spearman']:
                    expected = saved['outputs'][output].get(key)
                    if key == 'spearman':
                        expected = expected['value']
                    if expected is not None and not math.isclose(metrics[key], expected, abs_tol=1e-12, rel_tol=0):
                        raise ValueError(f'{model}/{variant}/{output}: {key} mismatch')
                rows.append(dict(model=model, variant=variant, output=output, n=len(frame), **metrics))
            best_rho = max(history, key=lambda h: float(h['dev_spearman']))
            best_rmse = min(history, key=lambda h: float(h['dev_rmse']))
            if int(best_rho['epoch']) != meta['best_epoch']:
                raise ValueError('Selected checkpoint is not the first max-Spearman epoch')
            exported_rmse = statistics.mean(saved['outputs'][o]['rmse'] for o in ['ModAvg','HeadAvg'])
            if not math.isclose(exported_rmse, float(best_rho['dev_rmse']), abs_tol=1e-12, rel_tol=0):
                raise ValueError('Selected history RMSE differs from exported predictions')
            epochs.append(dict(model=model, variant=variant, epochs_run=len(history),
                selected_epoch=int(best_rho['epoch'])+1, selected_mean_rmse=float(best_rho['dev_rmse']),
                selected_mean_spearman=float(best_rho['dev_spearman']),
                min_rmse_epoch=int(best_rmse['epoch'])+1, min_mean_rmse=float(best_rmse['dev_rmse']),
                mean_spearman_at_min_rmse=float(best_rmse['dev_spearman'])))
            sources.append(dict(model=model, variant=variant, file=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), packages=meta['packages'], python=meta['python']))
    return rows, epochs, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', required=True, type=Path)
    parser.add_argument('--bias-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    rows, epochs, sources = audit(args.baseline_dir, args.bias_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('error_decomposition', rows), ('checkpoint_history', epochs)]:
        with (args.output / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    (args.output / 'provenance.json').write_text(json.dumps(dict(sources=sources,
        limits=['Existing seed42 dev runs only; no new GPU experiment',
                'Centered RMSE uses dev gold; diagnostic, not a deployable prediction',
                'History mean RMSE is arithmetic mean of two output RMSEs; per-output history unavailable']), indent=2)+'\n')
    print(json.dumps(epochs, indent=2))


if __name__ == '__main__':
    main()
