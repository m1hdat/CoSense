"""Verify supplied bias artifacts and compare NN dev runs with paired baselines."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--bias-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--saved-versions', type=Path, default=Path(__file__).with_name('SAVED_VERSIONS.json'))
    args = parser.parse_args()
    import pandas as pd
    from compartment.artifacts import read_bundle
    from compartment.metrics import evaluate
    from diagnose import diagnose
    args.output.mkdir(parents=True, exist_ok=False)
    saved_versions = json.loads(args.saved_versions.read_text()) if args.saved_versions.is_file() else {}
    verification = {}
    # Verify full archives without extracting or executing checkpoint contents.
    for model in ['B1', 'B2']:
        tag = f'NN_{model}_seed42_full'
        inventory = json.loads((args.bias_dir / f'{tag}_ARTIFACTS.json').read_text())
        require((inventory['model'], inventory['task'], inventory['seed'], inventory['smoke']) ==
                (model, 'nn', 42, False), 'Artifact identity mismatch')
        require(inventory['status'] == 'OUTPUTS_READY', 'Artifacts not ready')
        checked = []
        for item in inventory['files']:
            require(Path(item['name']).name == item['name'], 'Invalid artifact filename')
            path = args.bias_dir / item['name']
            actual_hash = sha256(path)
            require(path.stat().st_size == item['bytes'] and actual_hash == item['sha256'],
                    f'Artifact hash/size mismatch: {path}')
            with zipfile.ZipFile(path) as archive:
                require(archive.testzip() is None, f'CRC failure: {path}')
                if item['role'] == 'complete':
                    required = ['best.pt', 'resume.pt', 'config.json', 'encoder_config/config.json',
                                'tokenizer/tokenizer.json', 'dev/results_bundle.zip']
                    require(set(required) <= set(archive.namelist()), 'Missing model files')
                    results = args.bias_dir / f'{tag}_results.zip'
                    require(archive.read('dev/results_bundle.zip') == results.read_bytes(), 'Embedded bundle mismatch')
                    with zipfile.ZipFile(results) as small:
                        require(archive.read('dev/predictions.tsv') == small.read('predictions.tsv'), 'Embedded predictions mismatch')
                    require(not any(n.startswith('test/') for n in archive.namelist()), 'Unexpected test exports')
            checked.append(dict(role=item['role'], path=str(path), sha256=actual_hash,
                                bytes=path.stat().st_size, crc='passed'))
        require({x['role'] for x in checked} == {'complete', 'results'}, 'Missing artifact roles')
        verification[model] = dict(artifacts=checked, weight_inference='not performed', saved_version_url=None)
        if model in saved_versions:
            version = saved_versions[model]
            verification[model].update(saved_version_url=version['url'],
                saved_version_id=version['script_version_id'], saved_version_source=version['source'],
                saved_version_remote_verification=version['remote_verification'])
        print(f'{model}: artifact hashes, CRC and embedded results passed', flush=True)

    metrics_rows, diagnostic_rows, run_rows = [], [], []
    frames = {}
    manifests = []
    for model in ['B1', 'B2']:
        for variant, directory in [('baseline', args.baseline_dir), ('bias', args.bias_dir)]:
            path = directory / f'NN_{model}_seed42_full_results.zip'
            meta, stored, frame = read_bundle(path)
            require((meta['task'], meta['model'], meta['seed']) == ('nn', model, 42), 'Run identity mismatch')
            for column, expected in [('task', 'nn'), ('model', model), ('seed', 42), ('split', 'dev')]:
                require(set(frame[column]) == {expected}, f'Unexpected {column}')
            require(meta['config']['smoke'] is False, 'Smoke run supplied')
            require(bool(meta['config'].get('init_train_mean_bias')) == (variant == 'bias'), 'Bias mode mismatch')
            computed, _ = evaluate(frame, 'nn')
            for output, values in computed['outputs'].items():
                require(values['n_valid'] == len(frame) == 504, 'Invalid prediction coverage')
                for key in ['rmse', 'mae', 'delta_mae_macro']:
                    if key in stored['outputs'][output]:
                        require(abs(stored['outputs'][output][key] - values[key]) < 1e-12, f'Metric mismatch: {key}')
                for key in ['spearman', 'context_variance']:
                    old, new = stored['outputs'][output][key]['value'], values[key]['value']
                    require(old is None and new is None or old is not None and new is not None and abs(old-new) < 1e-12, f'Metric mismatch: {key}')
                metrics_rows.append(dict(model=model, variant=variant, output=output,
                    spearman=values['spearman']['value'], mae=values['mae'], rmse=values['rmse'],
                    delta_mae=values['delta_mae_macro'], variance_correlation=values['context_variance']['value']))
            diagnostic_rows.extend(dict(model=model, variant=variant, **r) for r in diagnose(frame))
            with zipfile.ZipFile(path) as archive:
                require(archive.testzip() is None, 'Results CRC failed')
                manifest = json.loads(archive.read('split_manifest.json'))
                history = pd.read_csv(io.BytesIO(archive.read('training_history.csv')))
            require(set(frame.context_id) == set(manifest['partitions']['dev']), 'Dev IDs do not match manifest')
            require(set(meta['actual_split_ids']) == {'train', 'dev'}, 'Unexpected evaluated splits')
            require(all(ids == manifest['partitions'][split] for split, ids in meta['actual_split_ids'].items()), 'Actual split IDs mismatch')
            manifests.append(manifest)
            frames[(model, variant)] = (meta, frame.set_index('context_id').sort_index())
            run_rows.append(dict(model=model, variant=variant, source=str(path), sha256=sha256(path),
                best_epoch=int(meta['best_epoch'])+1, epochs_run=len(history),
                elapsed_seconds=meta.get('elapsed_seconds_this_invocation'), packages=meta['packages'],
                python=meta['python'], device=meta['device'], preprocessing=meta['preprocessing']))
    require(all(m == manifests[0] for m in manifests), 'Frozen manifests differ')
    reference = frames[('B1', 'baseline')][1]
    columns = ['task', 'target', 'context', 'part1', 'part2', 'gold_ModAvg', 'gold_HeadAvg']
    for meta, frame in frames.values():
        require(reference[columns].equals(frame[columns]), 'ID/gold/context differs')
    for model in ['B1', 'B2']:
        old, new = [frames[(model, variant)][0] for variant in ['baseline', 'bias']]
        differences = {k: [old['config'].get(k), new['config'].get(k)]
                       for k in old['config'].keys() | new['config'].keys()
                       if old['config'].get(k) != new['config'].get(k)}
        require(set(differences) == {'init_train_mean_bias', 'cuong_decision'}, 'Unexpected recipe change')
        verification[model].update(config_differences=differences,
            environment_matches={k: old.get(k) == new.get(k) for k in ['packages', 'python', 'device']})
    table = pd.DataFrame(metrics_rows)
    delta = table.pivot(index=['model', 'output'], columns='variant', values=['spearman', 'mae', 'rmse', 'delta_mae', 'variance_correlation'])
    changes = pd.DataFrame({key: delta[key]['bias']-delta[key]['baseline'] for key in ['spearman', 'mae', 'rmse', 'delta_mae', 'variance_correlation']}).reset_index()
    table.to_csv(args.output / 'metrics.csv', index=False)
    changes.to_csv(args.output / 'changes_bias_minus_baseline.csv', index=False)
    pd.DataFrame(diagnostic_rows).to_csv(args.output / 'diagnostics.csv', index=False)
    (args.output / 'verification.json').write_text(json.dumps(dict(models=verification, runs=run_rows,
        checks='CRC, hashes, embedded results, metrics, manifests, actual split IDs and paired ID/gold/context passed',
        limitations=['No raw training data or executed notebooks supplied', 'Checkpoint loading/inference/resume not performed', 'Single seed; dev only']), indent=2))
    print(table.to_string(index=False))
    print('\nBias minus baseline:\n' + changes.to_string(index=False))


if __name__ == '__main__':
    main()
