"""Notebook structure checks; no local training or pretrained downloads."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_run_without_local_source_or_generator():
    for variant in ['B1', 'B2']:
        nb = json.loads((ROOT / f'notebooks/{variant}_kaggle.ipynb').read_text(encoding='utf-8'))
        assert nb['metadata']['cosense']['standalone'] is True
        assert nb['metadata']['cosense']['model'] == variant
        for cell in nb['cells']:
            if cell['cell_type'] != 'code':
                continue
            tree = ast.parse(''.join(cell['source']))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert node.level == 0
                    assert not (node.module or '').startswith('compartment')
                if isinstance(node, ast.Import):
                    assert all(not alias.name.startswith('compartment') for alias in node.names)
