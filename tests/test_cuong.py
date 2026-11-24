"""Offline checks for diagnostic calculations and experiment preparation."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'handoff/week3_cuong/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_affine_diagnostic_recovers_known_shift_and_scale():
    y = np.array([1., 2., 3., 4.])
    frame = pd.DataFrame({f'{prefix}_{output}': values
        for output in ['ModAvg', 'HeadAvg']
        for prefix, values in [('gold', y), ('pred', 2*y+1)]})
    for row in load_script('diagnose').diagnose(frame):
        assert row['mean_error'] == pytest.approx(3.5)
        assert row['pred_gold_sd_ratio'] == pytest.approx(2)
        assert row['affine_dev_rmse'] == pytest.approx(0, abs=1e-12)
        assert row['affine_slope'] == pytest.approx(0.5)
        assert row['affine_intercept'] == pytest.approx(-0.5)
    frame['pred_ModAvg'] = 2.
    frame['gold_HeadAvg'] = 3.
    rows = load_script('diagnose').diagnose(frame)
    assert rows[0]['affine_slope'] == 0
    assert rows[1]['pred_gold_sd_ratio'] is None


@pytest.mark.parametrize('variant', ['B1', 'B2'])
def test_generated_bias_preserves_baseline_recipe(tmp_path, monkeypatch, variant):
    script = load_script('prepare_experiment')
    notebook = script.prepare(variant, 'offline fixture decision')
    config_cell = next(''.join(c['source']) for c in notebook['cells']
                       if c['cell_type'] == 'code' and 'CONFIG = dict(' in ''.join(c['source']))
    monkeypatch.delenv('OUTPUT_ROOT', raising=False)
    # Execute configuration in /tmp, emulating Kaggle without CUDA or downloads.
    config_cell = config_cell.replace('/kaggle/working', str(tmp_path))
    import os
    ns = dict(Path=Path, os=os)
    exec(config_cell, ns)
    config = ns['CONFIG']
    assert config['seed'] == ns['SPLIT_SEED'] == 42
    assert config['lr'] == 2e-5
    assert config['init_train_mean_bias'] is True
    assert ns['EVALUATE_LOCAL_TEST'] is False
    assert ns['RESUME'] is None
    assert all(not c.get('outputs') for c in notebook['cells'])


def test_bias_initialization_uses_only_finite_training_labels_and_skips_resume():
    script = load_script('prepare_experiment')
    class Bias:
        dtype = 'float'
        def copy_(self, values): self.values = values
    class Context:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    from types import SimpleNamespace
    bias = Bias()
    torch = SimpleNamespace(no_grad=Context, tensor=lambda values, **kwargs: values)
    ns = dict(np=np, torch=torch, config={'init_train_mean_bias': True}, resume_state=None,
        subsets={'train': [{'labels': [1., np.nan]}, {'labels': [3., 4.]}],
                 'dev': [{'labels': [999., 999.]}]},
        model=SimpleNamespace(head=[None, SimpleNamespace(bias=bias)]), device='cpu')
    import textwrap
    exec(textwrap.dedent(script.BIAS_INIT), ns)
    assert bias.values == [2., 4.]
    ns['resume_state'] = {'model': 'saved'}
    bias.values = 'unchanged'
    exec(textwrap.dedent(script.BIAS_INIT), ns)
    assert bias.values == 'unchanged'
