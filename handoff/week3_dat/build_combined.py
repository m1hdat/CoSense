"""Bundle unchanged training helpers into one sequential four-run notebook."""
import ast
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main(task='nn'):
    assert task in ('nn', 'pv')
    label = task.upper()
    b1 = json.loads((ROOT / f'notebooks/{label}_B1_kaggle.ipynb').read_text(encoding='utf-8'))
    b2 = json.loads((ROOT / f'notebooks/{label}_B2_kaggle.ipynb').read_text(encoding='utf-8'))
    for index in (2, 6, 7, 8):
        assert b1['cells'][index]['source'] == b2['cells'][index]['source']
    cells = []

    def add(kind, text):
        cell = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code':
            ast.parse(text)
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', f'''# Đạt tuần 3 — {label} B1/B2 × seed 43/44

**Một notebook, bốn lần train độc lập chạy tuần tự.** Dùng lại code của hai notebook {label} cũ; giữ nguyên split seed 42, backbone và recipe. Không chạy lại seed 42. {'PV nhiều seed là phần bổ sung theo yêu cầu; kế hoạch tuần 3 chỉ bắt buộc nhiều seed cho NN.' if task == 'pv' else 'Không chạy PV trong notebook này.'}

Kaggle: Add Input **cosense-kaggle-data**, bật GPU + Internet, **Save Version → Save & Run All**. Không cần tải thêm code/model. Mỗi run có complete ZIP, results ZIP và ARTIFACTS riêng; tải đủ bốn bộ từ Output của saved version. Model/ZIP của bốn run chiếm nhiều GB; giữ đủ dung lượng trong /kaggle/working.
''')
    add('code', ''.join(b1['cells'][2]['source']))
    config = ''.join(b1['cells'][4]['source'])
    config = config[:config.index("if TASK not in")]
    config = config.replace("NOTEBOOK_MODEL = 'B1'\nMODEL = 'B1'\n", '')
    config = config.replace('SEED = 42\n', 'SEED = 43\n', 1)
    config += f"\nassert TASK == '{task}' and SPLIT_SEED == 42 and not SMOKE\nassert RESUME is None and not EVALUATE_LOCAL_TEST\nprint('QUEUE: B1 seed43, B2 seed43, B1 seed44, B2 seed44; frozen split seed42')\n"
    add('code', config)
    for index in (6, 7, 8):
        add('code', ''.join(b1['cells'][index]['source']))
    add('code', '''import gc, tempfile
import pandas as pd
records, manifest, DATA_PROVENANCE = load_frozen_task(
    DATA_ROOT, TASK, Path(tempfile.gettempdir())/'cosense_input_cache', split_seed=SPLIT_SEED)
print(json.dumps(DATA_PROVENANCE, indent=2))
RUN_RESULTS = []

def run_one(variant, seed):
    config = dict(CONFIG, seed=seed)
    assert variant in ('B1', 'B2') and seed in (43, 44)
    assert not config['smoke'] and SPLIT_SEED == 42
    tag = f'{TASK.upper()}_{variant}_seed{seed}_full'
    run_root = OUTPUT_ROOT/tag
    marker = PUBLISH_ROOT/f'{tag}_ARTIFACTS.json'
    if marker.is_file():
        inventory = verify_published_artifacts(marker)
        assert (inventory['model'], inventory['task'], inventory['seed'], inventory['smoke']) == (variant, TASK, seed, False)
        saved = json.loads((run_root/'run_metadata.json').read_text())
        assert saved['config'] == config
        assert json.loads((run_root/'data_provenance.json').read_text()) == DATA_PROVENANCE
        assert json.loads((run_root/'split_manifest.json').read_text()) == manifest
        print('REUSE: already completed and verified', tag, flush=True)
    else:
        if run_root.exists() and any(run_root.iterdir()):
            raise RuntimeError(f'Incomplete run exists: {run_root}. Keep it; start a fresh Kaggle version/session to rerun, or recover explicitly. No automatic overwrite/resume.')
        run_root.mkdir(parents=True, exist_ok=True)
        dump(run_root/'data_provenance.json', DATA_PROVENANCE)
        print('TRAIN:', tag, flush=True)
        trained = tokenizer = metadata = None
        try:
            trained, tokenizer, metadata = train(
                records, manifest, config, variant, run_root,
                resume=None, evaluate_test=False)
            publish_run_artifacts(run_root, PUBLISH_ROOT, variant, TASK, seed, smoke=False)
        finally:
            del trained, tokenizer, metadata
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        verify_published_artifacts(marker)
    metrics = json.loads((run_root/'dev/metrics.json').read_text())
    predictions = pd.read_csv(run_root/'dev/predictions.tsv', sep='\t')
    for output, values in metrics['outputs'].items():
        RUN_RESULTS.append(dict(model=variant, seed=seed, output=output,
            spearman=values['spearman']['value'], rmse=values['rmse'],
            mae=float((predictions['gold_'+output]-predictions['pred_'+output]).abs().mean()),
            delta_mae=values['delta_mae_macro']))
    print('OUTPUTS_READY:', tag, flush=True)
''')
    for seed in (43, 44):
        for variant in ('B1', 'B2'):
            add('markdown', f'## Train {label} {variant} — seed {seed}\nModel mới độc lập; không resume seed khác.\n')
            add('code', f"run_one('{variant}', {seed})\n")
    add('markdown', '## Tổng hợp và downloads\nChỉ gồm seed 43/44. Ghép với seed 42 đã có để tính trung bình ± độ lệch chuẩn qua ba seed.\n')
    add('code', '''table = pd.DataFrame(RUN_RESULTS)
assert len(table) == 4 * len(SCHEMAS[TASK]['labels'])
table.to_csv(PUBLISH_ROOT/f'{TASK.upper()}_week3_seeds43_44_metrics.csv', index=False)
display(table)
for seed in (43, 44):
    for variant in ('B1', 'B2'):
        marker = PUBLISH_ROOT/f'{TASK.upper()}_{variant}_seed{seed}_full_ARTIFACTS.json'
        inventory = verify_published_artifacts(marker)
        print('VERIFIED:', variant, seed, [f['name'] for f in inventory['files']])
print('ALL_FOUR_RUNS_READY: download complete/results ZIPs + ARTIFACTS for all four runs from saved version Output')
''')
    notebook = dict(cells=cells, metadata=b1['metadata'], nbformat=4, nbformat_minor=5)
    for index, cell in enumerate(cells):
        cell['id'] = f'week3-dat-{index}'
    target = HERE / f'{label}_B1_B2_seeds43_44_kaggle.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(target.name, ': 4 sequential runs; training/data/metrics helpers identical to originals')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('nn', 'pv'), default='nn')
    main(parser.parse_args().task)
