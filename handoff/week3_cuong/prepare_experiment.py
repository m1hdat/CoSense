"""Generate separate Kaggle copies after documenting the diagnostic decision."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BIAS_INIT = '''    if config.get('init_train_mean_bias') and not resume_state:
        train_labels=np.asarray([r['labels'] for r in subsets['train']],dtype=float)
        means=[]
        for column in train_labels.T:
            finite=column[np.isfinite(column)]
            if not len(finite): raise ValueError('No finite train labels for bias initialization')
            means.append(float(finite.mean()))
        with torch.no_grad():
            model.head[1].bias.copy_(torch.tensor(means,device=device,dtype=model.head[1].bias.dtype))
'''


def prepare(variant, decision):
    notebook = json.loads((ROOT / f'notebooks/NN_{variant}_kaggle.ipynb').read_text())
    config_found = train_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        if 'CONFIG = dict(' in code:
            # Append after baseline config so all provenance/export cells receive it.
            code += "\nCONFIG['init_train_mean_bias'] = True\n"
            code += f"CONFIG['cuong_decision'] = {decision!r}\nTASK = 'nn'\nSEED = 42\nEVALUATE_LOCAL_TEST = False\n"
            config_found = True
        if 'def train(' in code:
            anchor = "    if config.get('smoke'):\n        subsets="
            assert code.count(anchor) == 1
            code = code.replace(anchor, BIAS_INIT + anchor)
            train_found = True
        # Isolate all working and published artifacts, including smoke/full runs.
        code = code.replace('/kaggle/working', '/kaggle/working/cuong_bias')
        ast.parse(code)
        cell['source'] = code.splitlines(keepends=True)
        cell['outputs'] = []
        cell['execution_count'] = None
    assert config_found and train_found
    notebook['cells'].insert(0, dict(cell_type='markdown', metadata={}, source=[
        f'# Cương — NN {variant}, bias, seed42\n',
        f'Decision: {decision}\n',
        'Run smoke first, then full. Keep frozen split42; do not evaluate test.\n',
        'Training has not been executed by this generator.\n']))
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', required=True, help='Diagnostic evidence for the bias experiment')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.decision.strip():
        parser.error('--decision must explain the diagnostic choice')
    args.output.mkdir(parents=True, exist_ok=False)
    for variant in ['B1', 'B2']:
        path = args.output / f'NN_{variant}_bias_seed42_kaggle.ipynb'
        path.write_text(json.dumps(prepare(variant, args.decision), ensure_ascii=False, indent=1) + '\n')
        print(path)


if __name__ == '__main__':
    main()
