"""Prepare optional two-run NN follow-up; never overwrite the historical pilot."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    notebook = json.loads((HERE / 'NN_B1_B2_seeds43_44_kaggle.ipynb').read_text(encoding='utf-8'))
    reference = json.loads((ROOT / 'results/nn/B1_seed43/run_metadata.json').read_text())
    cells = []
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if source.startswith('run_one(') and ', 44)' in source:
            continue
        if cell['cell_type'] == 'markdown' and source.startswith('## Train NN') and 'seed 44' in source:
            continue
        if source.startswith('# Đạt tuần 3'):
            source = '''# NN B1/B2 seed42 — kiểm soát môi trường

**Bổ sung tùy chọn, hai run** để so với seed43/44 trong cùng môi trường. Kết quả 12 run hiện có vẫn giữ nguyên. Import thành một notebook Kaggle mới, thêm Input `cosense-data`, bật GPU + Internet, Save & Run All. Không chạy tiếp trong session của notebook cũ.

Cell kiểm tra môi trường phải đạt trước khi train. Nếu không khớp, notebook dừng; không đổi công tắc để bỏ qua rồi gọi đây là đo nhiễu seed thuần. Mỗi model được khởi tạo mới, không resume. Giữ split42 và frozen dữ liệu. Khi tải về, lưu riêng ở `results/nn/B1_seed42_envmatch/` và `B2_seed42_envmatch/`, **không thay pilot seed42 cũ**.
'''
        elif source.startswith('## Train NN'):
            source = source.replace('seed 43', 'seed 42')
        elif source.startswith('## Tổng hợp'):
            source = '## Hai run đối chiếu\nGhép bản seed42 mới với seed43/44, giữ báo cáo pilot cũ riêng.\n'
        elif cell['cell_type'] == 'code':
            source = source.replace('SEED = 43\n', 'SEED = 42\n')
            source = source.replace("DATA_ROOT = Path(os.environ.get('DATA_ROOT', '/kaggle/input'))",
                                    "DATA_ROOT = Path('/kaggle/input/datasets/minhdat27/cosense-data')")
            source = source.replace("print('QUEUE: B1 seed43, B2 seed43, B1 seed44, B2 seed44; frozen split seed42')",
                                    "print('QUEUE: B1 seed42, B2 seed42; matched environment; frozen split seed42')")
            source = source.replace('seed in (43, 44)', 'seed in (42,)')
            source = source.replace(', 43)', ', 42)')
            source = source.replace('len(table) == 8', 'len(table) == 4')
            source = source.replace("NN_week3_seeds43_44_metrics.csv", "NN_seed42_envmatch_metrics.csv")
            source = source.replace('ALL_FOUR_RUNS_READY', 'BOTH_ENV_MATCHED_RUNS_READY')
            source = source.replace('for all four runs', 'for both seed42 runs')
        cell = dict(cell, source=source.splitlines(keepends=True))
        if cell['cell_type'] == 'code':
            ast.parse(source)
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    guard = f'''# Do not infer seed-only variation if the runtime image changed again.
import importlib.metadata, platform
EXPECTED_PACKAGES = {reference['packages']!r}
observed = {{name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}}
differences = {{name: (version, observed[name]) for name, version in EXPECTED_PACKAGES.items() if observed[name] != version}}
if platform.python_version() != {reference['python']!r}:
    differences['python'] = ({reference['python']!r}, platform.python_version())
if not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle GPU')
if torch.cuda.get_device_name(0) != {reference['device']!r}:
    differences['gpu'] = ({reference['device']!r}, torch.cuda.get_device_name(0))
if differences: raise RuntimeError(f'Environment differs from seed43/44 (expected, actual): {{differences}}. Stop and review image/packages before training.')
print('ENVIRONMENT_MATCHED:', observed, platform.python_version(), torch.cuda.get_device_name(0))
'''
    cells.insert(2, dict(cell_type='code', metadata={}, source=guard.splitlines(keepends=True),
                         outputs=[], execution_count=None))
    for index, cell in enumerate(cells):
        cell['id'] = f'nn-envmatch-{index}'
    notebook['cells'] = cells
    target = HERE / 'followup/NN_B1_B2_seed42_envmatch_kaggle.ipynb'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    print(target.relative_to(ROOT))


if __name__ == '__main__':
    main()
