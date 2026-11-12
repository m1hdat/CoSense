"""Verify real NN/PV results and describe three runs, including environment drift."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from compartment.data import SCHEMAS, load_records
from compartment.metrics import evaluate

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/week3_dat'
METRICS = ('spearman', 'rmse', 'mae', 'delta_mae', 'variance_correlation')
CHECKED_MODEL_ARCHIVES = set()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def same_number(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    else:
        assert np.isclose(actual, expected, atol=1e-12, rtol=0), (actual, expected)


def audit_run(task, model, seed, manifest, provenance, reference_meta, raw):
    run = ROOT / f'results/{task}/{model}_seed{seed}'
    meta = read(run / 'run_metadata.json')
    assert (meta['task'], meta['model'], meta['seed']) == (task, model, seed)
    assert not meta['config']['smoke'] and meta['config']['seed'] == seed
    assert {k: v for k, v in meta['config'].items() if k != 'seed'} == {
        k: v for k, v in reference_meta['config'].items() if k != 'seed'}
    assert read(run / 'data_provenance.json') == provenance
    assert read(run / 'split_manifest.json') == manifest
    for split in ('train', 'dev'):
        assert set(meta['actual_split_ids'][split]) == set(manifest['partitions'][split])
    frame = pd.read_csv(run / 'dev/predictions.tsv', sep='\t', dtype={'context_id': str})
    assert not frame.context_id.duplicated().any()
    assert set(frame.context_id) == set(manifest['partitions']['dev'])
    for column, value in [('task', task), ('model', model), ('seed', seed), ('split', 'dev')]:
        assert set(frame[column]) == {value}
    # Verify every row against immutable raw data, not just another model's output.
    labels = SCHEMAS[task]['labels']
    for item in frame.to_dict('records'):
        record = raw[item['context_id']]
        for key in ('target', 'context', 'part1', 'part2', 'alignment_status'):
            assert item[key] == record[key]
        for index, output in enumerate(labels):
            same_number(item['gold_'+output], record['labels'][index])
    assert np.isfinite(frame[[c+o for o in labels for c in ('gold_', 'pred_')]].to_numpy(float)).all()
    computed, _ = evaluate(frame, task)
    saved = read(run / 'dev/metrics.json')
    values = []
    for output, m in computed['outputs'].items():
        assert m['n_valid'] == m['n_total'] == len(frame)
        expected = saved['outputs'][output]
        for key in ('spearman', 'context_variance'):
            same_number(m[key]['value'], expected[key]['value'])
        for key in ('rmse', 'delta_mae_macro'):
            same_number(m[key], expected[key])
        values.append(dict(task=task, model=model, seed=seed, output=output, rows=len(frame),
                           spearman=m['spearman']['value'], rmse=m['rmse'],
                           mae=float((frame['gold_'+output]-frame['pred_'+output]).abs().mean()),
                           delta_mae=m['delta_mae_macro'],
                           variance_correlation=m['context_variance']['value']))
    with zipfile.ZipFile(run / 'results.zip') as archive:
        assert archive.testzip() is None
        for name in ('run_metadata.json', 'metrics.json', 'predictions.tsv', 'split_manifest.json'):
            assert archive.read(name) == (run / 'dev' / name).read_bytes()
    artifact_manifest_received = (run / 'ARTIFACTS.json').is_file()
    if artifact_manifest_received:
        publication = read(run / 'ARTIFACTS.json')
        assert (publication['task'], publication['model'], publication['seed']) == (task, model, seed)
        exported = next(f for f in publication['files'] if f['role'] == 'results')
        assert (run / 'results.zip').stat().st_size == exported['bytes']
        assert hashlib.sha256((run / 'results.zip').read_bytes()).hexdigest() == exported['sha256']
    resume_available = (run / 'complete.zip').exists()
    checkpoint = resume_available
    if resume_available:
        with zipfile.ZipFile(run / 'complete.zip') as archive:
            assert archive.testzip() is None
            for name in ('best.pt', 'resume.pt', 'tokenizer/tokenizer.json', 'encoder_config/config.json'):
                assert archive.getinfo(name).file_size > 0
            assert archive.read('dev/predictions.tsv') == (run / 'dev/predictions.tsv').read_bytes()
    elif (run / 'MODEL_ARCHIVE.json').is_file():
        location = read(run / 'MODEL_ARCHIVE.json')
        path = ROOT / location['archive']
        assert path.resolve().is_relative_to(ROOT.resolve())
        with zipfile.ZipFile(path) as archive:
            if path not in CHECKED_MODEL_ARCHIVES:
                assert archive.testzip() is None
                CHECKED_MODEL_ARCHIVES.add(path)
            for name in ('best.pt', 'tokenizer/tokenizer.json', 'encoder_config/config.json'):
                assert archive.getinfo(location['prefix']+name).file_size > 0
            assert archive.read(location['prefix']+'dev/predictions.tsv') == (run/'dev/predictions.tsv').read_bytes()
        assert location['inference_16_rows_passed'] and not location['includes_resume']
        checkpoint = True
    history = pd.read_csv(run / 'dev/training_history.csv')
    seconds = meta.get('elapsed_seconds_this_invocation')
    if seconds is None:
        seconds = read(run / 'progress.json')['elapsed_seconds']
    environment_diff = {name: {'seed42': reference_meta['packages'].get(name),
                               'this_run': version}
                        for name, version in meta['packages'].items()
                        if version != reference_meta['packages'].get(name)}
    if meta['python'] != reference_meta['python']:
        environment_diff['python'] = {'seed42': reference_meta['python'], 'this_run': meta['python']}
    audit = dict(task=task, model=model, seed=seed, results_crc_ok=True,
                 artifact_manifest_received=artifact_manifest_received,
                 metrics_recomputed_match=True, raw_gold_context_match=True,
                 frozen_train_dev_split_match=True, config_matches_except_seed=True,
                 checkpoint_downloaded_and_crc_checked=checkpoint,
                 resume_state_available=resume_available,
                 environment_matches_seed42=not environment_diff,
                 environment_diff=environment_diff)
    info = read(run / 'IMPORT.json') if (run / 'IMPORT.json').exists() else {}
    runtime = dict(task=task, model=model, seed=seed, status='results_verified', gpu=meta['device'],
                   training_seconds=seconds, training_minutes=seconds/60,
                   epochs_run=len(history), best_epoch=int(meta['best_epoch'])+1,
                   checkpoint_local=checkpoint, saved_version_url=info.get('saved_version_url'),
                   resume_state_available=resume_available,
                   environment=json.dumps(meta['packages'], sort_keys=True), python=meta['python'],
                   dev_fallback=meta['preprocessing']['dev']['fallback'],
                   dev_rows=len(frame), dev_truncated=meta['preprocessing']['dev']['truncated'])
    return values, runtime, audit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, runs, audits, missing = [], [], [], []
    for task in ('nn', 'pv'):
        manifest = read(ROOT / f'data/splits/{task}/split_manifest.json')
        source = ROOT / f'data/train/en-{task}-train.tsv'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
        raw = {r['context_id']: r for r in load_records(source, task)}
        provenance = read(ROOT / f'results/{task}/B1_seed42/data_provenance.json')
        assert provenance['source_sha256'] == manifest['source_sha256'] and provenance['split_seed'] == 42
        for model in ('B1', 'B2'):
            reference = read(ROOT / f'results/{task}/{model}_seed42/run_metadata.json')
            for seed in (42, 43, 44):
                if not (ROOT / f'results/{task}/{model}_seed{seed}/dev/predictions.tsv').exists():
                    missing.append(dict(task=task, model=model, seed=seed))
                    continue
                result, run, audit = audit_run(task, model, seed, manifest, provenance, reference, raw)
                rows.extend(result); runs.append(run); audits.append(audit)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(runs).to_csv(OUT / 'runs.csv', index=False)
    summary = []
    for (task, model, output), group in metrics.groupby(['task', 'model', 'output']):
        complete = set(group.seed) == {42, 43, 44}
        env = [a['environment_matches_seed42'] for a in audits if a['task'] == task and a['model'] == model]
        row = dict(task=task, model=model, output=output, n_seeds=len(group), seeds=','.join(map(str, group.seed)),
                   all_expected_seeds=complete, same_environment=all(env),
                   interpretation='descriptive across runs; environment changed' if not all(env) else 'same environment')
        for name in METRICS:
            # Undefined variance correlations (B1) remain NA.
            finite = group[name].dropna()
            row[name+'_mean'] = float(finite.mean()) if complete and len(finite) == len(group) else None
            row[name+'_sd'] = float(finite.std(ddof=1)) if complete and len(finite) == len(group) else None
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'seed_summary.csv', index=False)
    for task in ('nn', 'pv'):
        summary[summary.task == task].to_csv(OUT / f'{task}_seed_summary.csv', index=False)
    paired = metrics[metrics.model == 'B2'].merge(metrics[metrics.model == 'B1'],
                on=['task', 'seed', 'output'], suffixes=('_b2', '_b1'), validate='one_to_one')
    changes = paired[['task', 'seed', 'output']].copy()
    changes['spearman_b2_minus_b1'] = paired.spearman_b2-paired.spearman_b1
    for name in ('rmse', 'mae', 'delta_mae'):
        changes[name+'_gain_b1_minus_b2'] = paired[name+'_b1']-paired[name+'_b2']
    changes.to_csv(OUT / 'paired_seed_changes.csv', index=False)
    paired_summary = changes.groupby(['task', 'output']).agg(
        n=('seed', 'count'), spearman_difference_mean=('spearman_b2_minus_b1', 'mean'),
        spearman_difference_sample_sd=('spearman_b2_minus_b1', 'std'),
        spearman_positive_seeds=('spearman_b2_minus_b1', lambda s: int((s > 0).sum())),
        rmse_gain_mean=('rmse_gain_b1_minus_b2', 'mean'),
        mae_gain_mean=('mae_gain_b1_minus_b2', 'mean'),
        delta_mae_gain_mean=('delta_mae_gain_b1_minus_b2', 'mean')).reset_index()
    paired_summary.to_csv(OUT / 'paired_seed_summary.csv', index=False)
    verification = dict(runs=audits, missing_results=missing,
        model_checkpoints_missing=[f"{r['task'].upper()}_{r['model']}_seed{r['seed']}"
                                  for r in runs if not r['checkpoint_local']],
        model_bundle_checks=read(OUT/'model_bundle_verification.json') if (OUT/'model_bundle_verification.json').exists() else [],
        inference_check=read(OUT / 'pv_b1_inference_verification.json'),
        limits=['Dev-selected checkpoints, no test evaluation.',
                'Seed42 vs seeds43/44 differ in packages and Python; SD is descriptive, not pure seed noise.',
                'Saved-version URLs missing; received four NN manifests, PV manifests still missing.',
                'New bundles retain inference models, not optimizer/RNG state for training resume.'])
    (OUT / 'audit.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    write_report(summary, changes, runs, verification)
    print(f'Verified {len(runs)} real runs; {len(rows)} output rows; {len(verification["model_checkpoints_missing"])} inference checkpoints missing.')
    print(summary[['task', 'model', 'output', 'spearman_mean', 'spearman_sd', 'rmse_mean']].to_string(index=False))


def write_report(summary, changes, runs, verification):
    table = []
    for task, output in [('nn', 'ModAvg'), ('nn', 'HeadAvg'), ('pv', 'Avg')]:
        for model in ('B1', 'B2'):
            row = summary[(summary.task == task) & (summary.model == model) & (summary.output == output)].iloc[0]
            table.append(f"| {task.upper()} {output} | {model} | {row.spearman_mean:.4f} ± {row.spearman_sd:.4f} | {row.rmse_mean:.4f} | {row.mae_mean:.4f} | {row.delta_mae_mean:.4f} |")
    report = '''# Report tuần 3: Đạt · PV và dao động giữa các run

**Kết luận:** PV B2 tốt hơn B1 cả thứ hạng lẫn sai số ở ba seed; NN ModAvg chưa cho thấy lợi ích B2 ổn định. Thống kê ba seed là mô tả vì môi trường seed42 khác seed43/44.

## 1. Đã chạy gì
Đủ 12 run NN/PV × B1/B2 × seeds42/43/44, Tesla T4. Tám run mới hoàn tất 11/11, không lỗi; seed42 dùng lại pilot 27/10. GPU/phút từng run: `results/week3_dat/runs.csv`; thiếu link saved version.

## 2. Kết quả dev (42/43/44)
Spearman: mean ± sample SD (`ddof=1`); sai số: mean ba run. NN: 504 dòng/67 cụm; PV: 231 dòng/24 cụm. Cùng raw data, frozen split, recipe.

| Task/output | Model | Spearman ↑ | RMSE ↓ | MAE ↓ | Delta MAE ↓ |
|---|---|---:|---:|---:|---:|
''' + '\n'.join(table) + '''

**PV:** B2 tăng Spearman cả ba seed (+0,0971/+0,1774/+0,1275), giảm RMSE/MAE/Delta MAE; khác NN seed42. Chưa đánh giá test.

**NN:** B2−B1 ModAvg: +0,0561/−0,0305/+0,0392; mean +0,0216, SD chênh lệch 0,0459. SD Spearman B1/B2: 0,0146/0,0364. Mức tăng seed42 ≈0,057 chưa ổn định, có lần đổi dấu. HeadAvg tăng Spearman cả ba run; Delta MAE B2 tệ hơn B1 ở cả hai output, cả ba seed.

## 3. Đã đổi gì
Giữ split seed42, alignment, BERT revision, hyperparameters; đổi training seed và mount. Image thực tế đổi Transformers 5.16.1→5.20.2, tokenizers 0.23.1→0.23.3, torch cu128→cu130, Python 3.13.15→3.13.16. SD trên không đo riêng nhiễu seed.

## 4. Kiểm chứng
Tám results ZIP đạt CRC; ID/gold/context khớp raw data/manifest; metrics tính lại khớp ≤1e-12. Đã nhận model của cả tám run mới trong hai ZIP gộp: CRC/metadata/predictions khớp; nạp từng model thật, dự đoán lại 16 dòng dev đạt ngưỡng 1e-5. PV B1 seed42 cũng đã đạt kiểm tra này. B2 fallback NN 14/504, PV 43/231; không truncation.

## 5. Còn thiếu khi bàn giao
Cần bổ sung hai link saved version; bốn manifest NN đã khớp hash. Model mới đủ cho inference/báo cáo; đã chủ động bỏ resume.pt nên không có optimizer/RNG để tiếp tục train. Smoke PV cũ chưa có bằng chứng riêng.

## 6. Tiếp theo
Nộp report với giới hạn môi trường đã ghi; không cần train thêm để báo cáo kết quả hiện có. Nếu cần đo riêng nhiễu seed, bổ sung NN seed42 dưới đúng môi trường seed43/44, giữ pilot cũ.
'''
    total = sum(r['training_seconds'] for r in runs)
    new = sum(r['training_seconds'] for r in runs if r['seed'] != 42)
    report += f'\nGiờ làm tay: chưa ghi nhận · GPU training: {total/3600:.3f} giờ (tám run mới: {new/3600:.3f} giờ), không gồm setup/export.\n'
    (HERE / 'week3_Dat.md').write_text(report, encoding='utf-8')
    team_report = ROOT / 'reports/week3/Dat.md'
    team_report.parent.mkdir(parents=True, exist_ok=True)
    team_report.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
