"""Import the eight explicit results ZIPs; never execute downloaded notebooks."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {'run_metadata.json', 'metrics.json', 'predictions.tsv',
            'target_statistics.tsv', 'error_analysis.tsv', 'training_history.csv',
            'split_manifest.json', 'error_groups.tsv'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_if_equal_or_absent(path, content):
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f'Existing different artifact; keep both and review: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)


def main(downloads):
    for task in ('nn', 'pv'):
        source = downloads / f'{task}-43-44.ipynb'
        notebook = json.loads(source.read_text(encoding='utf-8'))
        expected = json.loads((ROOT / f'handoff/week3_dat/{task.upper()}_B1_B2_seeds43_44_kaggle.ipynb').read_text(encoding='utf-8'))
        assert len(notebook['cells']) == len(expected['cells'])
        provenance = None
        marker_count = 0
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            code = ''.join(cell['source'])
            planned = ''.join(expected['cells'][index]['source'])
            if index == 2:
                planned = planned.replace("DATA_ROOT = Path(os.environ.get('DATA_ROOT', '/kaggle/input'))",
                                          "DATA_ROOT = Path('/kaggle/input/datasets/minhdat27/cosense-data')")
            assert code == planned, f'Unexpected code change in {source.name} cell {index}'
            streams = []
            for output in cell.get('outputs', []):
                assert output.get('output_type') != 'error', f'Notebook error: {source.name}'
                if output.get('output_type') == 'stream':
                    value = output['text']
                    streams.append(value if isinstance(value, str) else ''.join(value))
            text = ''.join(streams)
            marker_count += text.count('ALL_FOUR_RUNS_READY:')
            if 'records, manifest, DATA_PROVENANCE = load_frozen_task' in code:
                provenance = json.JSONDecoder().raw_decode(text.lstrip())[0]
        assert marker_count == 1 and provenance
        reference = json.loads((ROOT / f'results/{task}/B1_seed42/data_provenance.json').read_text())
        assert provenance == reference
        papermill = notebook['metadata'].get('papermill', {})
        assert papermill.get('exception') is None and papermill.get('end_time')
        target_notebook = ROOT / f'results/{task}/executed_seeds43_44.ipynb'
        write_if_equal_or_absent(target_notebook, source.read_bytes())
        for seed in (43, 44):
            for model in ('B1', 'B2'):
                path = downloads / f'{task.upper()}_{model}_seed{seed}_full_results.zip'
                with zipfile.ZipFile(path) as archive:
                    assert len(archive.namelist()) == len(REQUIRED) and set(archive.namelist()) == REQUIRED
                    assert sum(i.file_size for i in archive.infolist()) < 20_000_000
                    assert archive.testzip() is None
                    meta = json.loads(archive.read('run_metadata.json'))
                    assert (meta['task'], meta['model'], meta['seed']) == (task, model, seed)
                    assert not meta['config']['smoke']
                    manifest = json.loads(archive.read('split_manifest.json'))
                    assert manifest == json.loads((ROOT / f'data/splits/{task}/split_manifest.json').read_text())
                    run = ROOT / f'results/{task}/{model}_seed{seed}'
                    for name in sorted(REQUIRED):
                        write_if_equal_or_absent(run / 'dev' / name, archive.read(name))
                    for name in ('run_metadata.json', 'split_manifest.json'):
                        write_if_equal_or_absent(run / name, archive.read(name))
                    write_if_equal_or_absent(run / 'config.json', json.dumps(meta['config'], indent=2).encode())
                write_if_equal_or_absent(run / 'results.zip', path.read_bytes())
                write_if_equal_or_absent(run / 'data_provenance.json', json.dumps(provenance, indent=2).encode())
                info = dict(source_file=path.name, source_sha256=digest(path), bytes=path.stat().st_size,
                            notebook=str(target_notebook.relative_to(ROOT)), notebook_sha256=digest(source),
                            notebook_execution=papermill, code_changes=['DATA_ROOT mount only'],
                            provenance_source='Printed loader output in executed notebook; matches seed42 frozen provenance',
                            results_crc_ok=True, downloaded_checkpoint=False,
                            notebook_reports_all_four_exports=True, saved_version_url=None)
                write_if_equal_or_absent(run / 'IMPORT.json', json.dumps(info, indent=2).encode())
                print('Imported:', task, model, seed, '(results only; checkpoint not supplied)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloads', type=Path, default=Path.home() / 'Downloads')
    main(parser.parse_args().downloads)
