"""Offline checks for the checkpoint-selection control experiment."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'handoff/week4_cuong/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CheckpointAuditTests(unittest.TestCase):
    def test_decomposition_and_tied_ranks(self):
        audit = load('audit_rmse')
        values = audit.error_decomposition([1., 2., 3.], [3., 4., 5.])
        self.assertEqual(values['rmse'], 2.)
        self.assertEqual(values['mean_error'], 2.)
        self.assertEqual(values['centered_mse'], 0.)
        self.assertAlmostEqual(values['spearman'], 1.)
        self.assertEqual(audit.ranks([4., 1., 1., 3.]), [4., 1.5, 1.5, 3.])

    def test_generated_training_keeps_original_loss_optimizer_and_stopping(self):
        module = load('prepare_checkpoint_audit')
        for task, seed in [('nn', 43), ('nn', 44), ('pv', 43)]:
            notebook = module.prepare(task, seed)
            original = json.loads((ROOT / f'notebooks/{task.upper()}_B2_kaggle.ipynb').read_text())
            def functions(nb):
                found = {}
                for cell in nb['cells']:
                    if cell['cell_type'] == 'code':
                        code = ''.join(cell['source'])
                        found.update({n.name: n for n in ast.parse(code).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))})
                return found
            before, after = functions(original), functions(notebook)
            for name in before.keys() - {'train'}:
                self.assertEqual(ast.dump(before[name]), ast.dump(after[name]), name)
            # Removing only instrumentation must recover the original training function.
            train_cell = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'def train(' in ''.join(c['source']))
            for block in [module.SETUP, module.TRACK, module.EXPORT]:
                self.assertEqual(train_cell.count(block), 1)
                train_cell = train_cell.replace(block, '')
            recovered = next(n for n in ast.parse(train_cell).body if isinstance(n, ast.FunctionDef) and n.name=='train')
            self.assertEqual(ast.dump(recovered), ast.dump(before['train']))
            config = next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'CONFIG = dict(' in ''.join(c['source']))
            self.assertIn(f'\nSEED = {seed}\n', config)
            self.assertIn('SPLIT_SEED = 42', config)
            self.assertIn('EVALUATE_LOCAL_TEST = False', config)
            self.assertNotIn('init_train_mean_bias', config)
            self.assertTrue(all(not c.get('outputs') for c in notebook['cells']))

    def test_rmse_tracking_uses_first_minimum_independently_of_spearman(self):
        module = load('prepare_checkpoint_audit')
        saved = []
        ns = dict(copy=copy, audit_epochs=[], audit_best_rmse=float('inf'), audit_best_epoch=None,
                  audit_directory=Path('/unused'), model=SimpleNamespace(state_dict=lambda: 'weights'),
                  torch=SimpleNamespace(save=lambda value, path: saved.append(value)))
        for epoch, rmse, rho in [(0, .8, .5), (1, .7, .4), (2, .7, .6), (3, .9, .9)]:
            ns.update(epoch=epoch, rmse=rmse, score=rho, metrics={'outputs': {'Avg': {'rmse': rmse}}})
            exec(textwrap.dedent(module.TRACK), ns)
        self.assertEqual(ns['audit_best_epoch'], 1)
        self.assertEqual([x['epoch'] for x in saved], [0, 1])
        self.assertEqual(len(ns['audit_epochs']), 4)

    def test_alternate_export_restores_control_model_and_metadata(self):
        module = load('prepare_checkpoint_audit')
        writes, exports, loads = {}, [], []
        output = Path('/unused')
        model = SimpleNamespace(load_state_dict=lambda state: loads.append(state),
            encoder=SimpleNamespace(config=SimpleNamespace(save_pretrained=lambda path: None)))
        ns = dict(copy=copy, output=output, audit_directory=output/'rmse_checkpoint',
            metadata={'best_epoch': 4, 'config': {'seed': 43}}, audit_best_epoch=1, audit_best_rmse=.7,
            config={'batch_size': 8, 'seed': 43}, variant='B2', manifest={}, history=[], audit_epochs=[],
            model=model, device='cpu', tokenizer=SimpleNamespace(save_pretrained=lambda path: None),
            torch=SimpleNamespace(load=lambda path, **kw: {'model': 'rmse' if 'rmse_checkpoint' in str(path) else 'control'}),
            subsets={'dev': ['record']}, datasets={'dev': SimpleNamespace(logs=[])},
            predict=lambda *args: 'predictions', prediction_frame=lambda *args: 'frame',
            dump=lambda path, value: writes.update({str(path): copy.deepcopy(value)}),
            export_bundle=lambda path, frame, meta, *args: exports.append(copy.deepcopy(meta)))
        exec(textwrap.dedent(module.EXPORT), ns)
        self.assertEqual(loads, ['rmse', 'control'])
        self.assertEqual(ns['metadata']['best_epoch'], 4)
        self.assertEqual(exports[0]['best_epoch'], 1)
        self.assertEqual(exports[0]['selection_policy'], 'mean_dev_rmse')


class SelectionCriteriaTests(unittest.TestCase):
    def test_mean_rmse_improvement_does_not_hide_head_regression(self):
        compare = load('compare_selectors')
        def metric(rmse, rho):
            return {'rmse': rmse, 'spearman': {'value': rho}}
        control = {'ModAvg': metric(.93, .43), 'HeadAvg': metric(.73, .60)}
        alternative = {'ModAvg': metric(.89, .44), 'HeadAvg': metric(.74, .58)}
        conditions = compare.assess_nn(control, alternative)
        self.assertFalse(conditions['rmse_decreases_both'])
        self.assertTrue(conditions['modifier_rmse_at_most_0_938'])
        self.assertTrue(conditions['spearman_drop_at_most_0_02'])
        self.assertFalse(all(conditions.values()))

    def test_success_requires_all_three_conditions(self):
        compare = load('compare_selectors')
        control = {o: {'rmse': 1., 'spearman': {'value': .6}} for o in ['ModAvg', 'HeadAvg']}
        alternative = {o: {'rmse': .9, 'spearman': {'value': .59}} for o in ['ModAvg', 'HeadAvg']}
        self.assertTrue(all(compare.assess_nn(control, alternative).values()))


if __name__ == '__main__':
    unittest.main()
