"""Verify full Kaggle selection artifacts and compare paired dev checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def assess_nn(control, alternative):
    return dict(
        rmse_decreases_both=all(alternative[o]['rmse'] < control[o]['rmse'] for o in ['ModAvg', 'HeadAvg']),
        modifier_rmse_at_most_0_938=alternative['ModAvg']['rmse'] <= .938,
        spearman_drop_at_most_0_02=all(alternative[o]['spearman']['value'] >= control[o]['spearman']['value']-.02 for o in ['ModAvg', 'HeadAvg']))


def same_number(a, b):
    return a is None and b is None or a is not None and b is not None and math.isclose(a, b, abs_tol=1e-12, rel_tol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--task', choices=['nn', 'pv'], default='nn')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import pandas as pd
    from compartment.artifacts import read_bundle
    from compartment.metrics import evaluate
    tag = f'{args.task.upper()}_B2_seed{args.seed}_full'
    expected_files = {f'{tag}_{suffix}.zip' for suffix in ['complete', 'results', 'RMSE_results']}
    inventory = json.loads((args.directory / f'{tag}_SELECTION_ARTIFACTS.json').read_text())
    require((inventory['task'], inventory['model'], inventory['seed'], inventory['smoke']) == (args.task, 'B2', args.seed, False), 'Unexpected inventory identity')
    require({i['name'] for i in inventory['files']} == expected_files and len(inventory['files'])==3, 'Unexpected artifact filenames')
    verified = {}
    for item in inventory['files']:
        path = args.directory / item['name']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(path.stat().st_size == item['bytes'] and digest == item['sha256'], f'Hash/size mismatch: {path}')
        with zipfile.ZipFile(path) as archive:
            require(archive.testzip() is None, f'CRC failed: {path}')
        verified[item['name']] = dict(sha256=digest, bytes=path.stat().st_size, crc='passed')
        print('Verified:', path.name, flush=True)
    original = json.loads((args.directory / f'{tag}_ARTIFACTS.json').read_text())
    require((original['task'],original['model'],original['seed'],original['smoke'],original['status']) == (args.task,'B2',args.seed,False,'OUTPUTS_READY'), 'Original inventory identity mismatch')
    require({i['name'] for i in original['files']} == expected_files-{f'{tag}_RMSE_results.zip'}, 'Original inventory files mismatch')
    for item in original['files']:
        require(verified[item['name']]['sha256']==item['sha256'] and verified[item['name']]['bytes']==item['bytes'], 'Inventory disagreement')
    frames, metas, outputs, manifests, metric_rows = {}, {}, {}, {}, []
    for policy, suffix in [('spearman', 'results'), ('rmse', 'RMSE_results')]:
        path = args.directory / f'{tag}_{suffix}.zip'
        meta, stored, frame = read_bundle(path)
        require((meta['task'],meta['model'],meta['seed'],meta['config']['smoke'])==(args.task,'B2',args.seed,False), 'Run identity mismatch')
        for col, value in [('task',args.task),('model','B2'),('seed',args.seed),('split','dev')]:
            require(set(frame[col])=={value}, f'Unexpected {col}')
        computed, _ = evaluate(frame, args.task)
        for output, values in computed['outputs'].items():
            require(values['n_valid']==len(frame), 'Nonfinite gold/predictions')
            for key in ['rmse', 'mae', 'delta_mae_macro']:
                require(same_number(values[key], stored['outputs'][output][key]), f'Metric mismatch: {key}')
            for key in ['spearman','context_variance']:
                require(same_number(values[key]['value'], stored['outputs'][output][key]['value']), f'Metric mismatch: {key}')
            metric_rows.append(dict(policy=policy, epoch=meta['best_epoch']+1, output=output,
                spearman=values['spearman']['value'], rmse=values['rmse'], mae=values['mae'],
                delta_mae=values['delta_mae_macro'], variance_correlation=values['context_variance']['value']))
        with zipfile.ZipFile(path) as archive:
            manifests[policy]=json.loads(archive.read('split_manifest.json'))
        frames[policy]=frame.set_index('context_id').sort_index()
        metas[policy]=meta; outputs[policy]=computed['outputs']
    manifest=manifests['spearman']
    require(manifest==manifests['rmse'], 'Manifest mismatch')
    for policy in ['spearman','rmse']:
        meta=metas[policy]
        require(set(frames[policy].index)==set(manifest['partitions']['dev']), 'Dev coverage mismatch')
        require(set(meta['actual_split_ids'])=={'train','dev'} and all(ids==manifest['partitions'][split] for split,ids in meta['actual_split_ids'].items()), 'Actual split IDs mismatch')
    for field in ['config','packages','python','device','actual_split_ids','preprocessing']:
        require(metas['spearman'][field]==metas['rmse'][field], f'Runtime mismatch: {field}')
    gold=[c for c in frames['spearman'] if c.startswith('gold_')]
    cols=['task','target','context','part1','part2','spans']+gold
    require(frames['spearman'][cols].equals(frames['rmse'][cols]), 'Paired ID/gold/context/spans mismatch')
    require(metas['rmse'].get('selection_policy')=='mean_dev_rmse' and metas['rmse'].get('shared_training_trajectory') is True and metas['rmse'].get('early_stopping_policy')=='mean_dev_spearman', 'Wrong selector policy')
    with zipfile.ZipFile(args.directory/f'{tag}_complete.zip') as archive:
        for policy, prefix, suffix in [('spearman','','results'),('rmse','rmse_checkpoint/','RMSE_results')]:
            for member in ['best.pt','tokenizer/tokenizer.json','encoder_config/config.json','config.json']:
                require(prefix+member in archive.namelist(), f'Missing checkpoint member: {prefix+member}')
            require(archive.read(prefix+'dev/results_bundle.zip')==(args.directory/f'{tag}_{suffix}.zip').read_bytes(), 'Embedded results mismatch')
        require(not any('/test/' in '/'+n for n in archive.namelist()), 'Unexpected test export')
        epochs=json.loads(archive.read('epoch_metrics.json'))
        selection=json.loads(archive.read('selection_comparison.json'))
    rho_epoch=max(epochs, key=lambda e:e['mean_spearman'])['epoch']
    rmse_epoch=min(epochs, key=lambda e:e['mean_rmse'])['epoch']
    require(metas['spearman']['best_epoch']==rho_epoch and metas['rmse']['best_epoch']==rmse_epoch, 'Selected epochs are not extrema')
    require(selection['spearman_epoch']==rho_epoch+1 and selection['rmse_epoch']==rmse_epoch+1, 'Selection summary mismatch')
    for policy in ['spearman','rmse']:
        selected=next(e for e in epochs if e['epoch']==metas[policy]['best_epoch'])
        for output, values in outputs[policy].items():
            require(same_number(values['rmse'],selected['outputs'][output]['rmse']) and same_number(values['spearman']['value'],selected['outputs'][output]['spearman']['value']), 'Epoch metrics do not match export')
    table=pd.DataFrame(metric_rows)
    changes=[]
    for output in outputs['spearman']:
        a=next(r for r in metric_rows if r['policy']=='spearman' and r['output']==output)
        b=next(r for r in metric_rows if r['policy']=='rmse' and r['output']==output)
        changes.append(dict(output=output, **{k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in ['spearman','rmse','mae','delta_mae','variance_correlation']}))
    criteria=assess_nn(outputs['spearman'],outputs['rmse']) if args.task=='nn' else {}
    version_file=Path(__file__).with_name('SAVED_VERSIONS.json')
    versions=json.loads(version_file.read_text()) if version_file.is_file() else {}
    saved_version=versions.get(f'{args.task}_B2_seed{args.seed}')
    summary=dict(task=args.task, seed=args.seed, rows=len(frames['spearman']), targets=frames['spearman'].target.nunique(),
        epochs_run=len(epochs), selection=selection, criteria=criteria,
        same_selected_epoch=rho_epoch==rmse_epoch,
        identical_predictions=frames['spearman'][['pred_'+o for o in outputs['spearman']]].equals(frames['rmse'][['pred_'+o for o in outputs['rmse']]]),
        meets_nn_criteria=all(criteria.values()) if criteria else None,
        elapsed_seconds=metas['spearman']['elapsed_seconds_this_invocation'],
        runtime={k:metas['spearman'][k] for k in ['python','packages','device','preprocessing']},
        saved_version=saved_version,
        caveats=['One seed, dev only', 'Checkpoint inference and executed notebook code not verified']
                + ([] if saved_version else ['No Kaggle saved version URL supplied']))
    args.output.mkdir(parents=True,exist_ok=False)
    table.to_csv(args.output/'metrics.csv',index=False)
    pd.DataFrame(changes).to_csv(args.output/'changes_rmse_minus_spearman.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'verification.json').write_text(json.dumps(dict(source=str(args.directory),artifacts=verified,paired_checks='passed',weight_inference='not performed',saved_version=saved_version),indent=2)+'\n')
    print(table.to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
