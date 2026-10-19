"""Meaningful data-boundary/artifact checks; no downloads or optimizer steps."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import kaggle_en_nn as pipeline
from package_kaggle_en_nn import build


class FakeTokenizer:
    def __call__(self, text, text_pair=None, **kwargs):
        if set(kwargs) != {'padding', 'truncation', 'return_token_type_ids'}:
            raise AssertionError('Unexpected tokenizer arguments')
        words = text.split() + (text_pair.split() if text_pair else [])
        ids = [101] + list(range(1000, 1000 + len(words))) + [102]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


class KaggleArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.archive = cls.base / 'cosense-en-nn-data.zip'
        with contextlib.redirect_stdout(io.StringIO()):
            cls.verification = build(cls.archive)
        cls.root = cls.base / 'expanded'
        with zipfile.ZipFile(cls.archive) as archive:
            archive.extractall(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_allowlist_has_no_credentials_or_unrelated_data(self):
        with zipfile.ZipFile(self.archive) as archive:
            names = set(archive.namelist())
        self.assertEqual(names, set(pipeline.payload_paths()) | {'bundle_manifest.json'})
        self.assertFalse(any('en-pv-train' in name or '.env' in name or '.git/' in name for name in names))
        self.assertFalse(self.verification['uploaded'])

    def test_both_settings_load_and_final_is_not_exported(self):
        expected = {'unseen_compound': {'train': 2444, 'validation': 523, 'final': 523},
                    'seen_context': {'train': 2443, 'validation': 524, 'final': 523}}
        for setting in pipeline.SETTINGS:
            selected, integrity = pipeline.load_checked(self.root, setting)
            self.assertEqual(set(selected), {'train', 'validation'})
            self.assertEqual(integrity['counts'], expected[setting])
            self.assertFalse(integrity['final_evaluated'])
            with tempfile.TemporaryDirectory() as temp:
                b0 = pipeline.save_prepared(selected, integrity, temp)
                self.assertEqual(b0['split'], 'validation')
                _, final = pipeline.read_csv(self.root / f'data/splits/en_nn/{setting}/final.csv')
                final_keys = {member['row_key'] for member in final}
                for variant in ('B1', 'B2'):
                    for split in ('train', 'validation'):
                        path = Path(temp) / 'prepared' / variant / f'{split}.jsonl'
                        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                        self.assertEqual(len(records), expected[setting][split])
                        self.assertFalse({r['metadata']['row_key'] for r in records} & final_keys)
                        self.assertEqual(set(records[0]['inputs']), {'text'} if variant == 'B1' else {'text', 'text_pair'})
                        self.assertEqual(len(records[0]['labels']), 2)
                self.assertFalse(list(Path(temp).rglob('final.*')))
            target = 0.6700543438908586 if setting == 'unseen_compound' else 0.7453409997844095
            self.assertAlmostEqual(b0['metrics']['compound_macro_two_output_mae'], target)

    def test_b1_ignores_context_and_forbidden_features(self):
        selected, _ = pipeline.load_checked(self.root, 'unseen_compound')
        row = selected['train'][0]
        changed = copy.deepcopy(row)
        changed.update(Context='unrelated sentence', ContextID='bad', ModAvg='0', HeadAvg='0', ModStd='999')
        self.assertEqual(pipeline.model_inputs(row, 'B1'), pipeline.model_inputs(changed, 'B1'))
        self.assertNotEqual(pipeline.model_inputs(row, 'B2'), pipeline.model_inputs(changed, 'B2'))
        self.assertEqual(pipeline.model_inputs(row, 'B2'), {'text': row['Compound'], 'text_pair': row['Context']})

    def test_tokenizer_preserves_rows_and_rejects_silent_truncation(self):
        selected, _ = pipeline.load_checked(self.root, 'unseen_compound')
        rows = selected['train'][:7]
        tokens, stats = pipeline.tokenize_rows(rows, 'B2', FakeTokenizer(), 256)
        self.assertEqual(len(tokens), 7)
        self.assertEqual(stats['truncated'], 0)
        self.assertEqual(set(tokens[0]), {'input_ids', 'attention_mask', 'labels'})
        self.assertEqual(tokens[0]['labels'], [float(rows[0]['ModAvg']), float(rows[0]['HeadAvg'])])
        with self.assertRaisesRegex(ValueError, 'no automatic cropping'):
            pipeline.tokenize_rows(rows, 'B2', FakeTokenizer(), 2)

    def test_changed_source_is_blocked(self):
        path = self.root / pipeline.SOURCE
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'Missing/changed'):
                pipeline.load_checked(self.root, 'unseen_compound')
        finally:
            path.write_bytes(original)

    def test_nested_mount_and_retained_zip_discovery(self):
        found = pipeline.discover_bundle(self.root, self.base / 'work')
        self.assertEqual(found, self.root)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'datasets/username/slug/versions/1'
            folder.mkdir(parents=True)
            (folder / self.archive.name).write_bytes(self.archive.read_bytes())
            found = pipeline.discover_bundle(temp, Path(temp) / 'work')
            self.assertTrue((found / pipeline.SOURCE).is_file())
            pipeline.verify_bundle(found)

    def test_unsafe_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            archive_path = Path(temp) / 'cosense-en-nn-data.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('../outside.txt', 'bad')
            with self.assertRaisesRegex(ValueError, 'unexpected'):
                pipeline.discover_bundle(temp, Path(temp) / 'work')
            self.assertFalse((Path(temp) / 'outside.txt').exists())

    def test_prediction_clipping_and_shape_guards(self):
        rows = [{'Compound': 'x', 'ModAvg': '0', 'HeadAvg': '5'}]
        scores, clipped = pipeline.score_predictions(rows, [[-1, 6]])
        self.assertEqual(clipped, [[0.0, 5.0]])
        self.assertEqual(scores['compound_macro_two_output_mae'], 0)
        self.assertEqual(scores['out_of_range_fraction'], 1)
        with self.assertRaises(ValueError):
            pipeline.score_predictions(rows, [[float('nan'), 1]])
        with self.assertRaises(ValueError):
            pipeline.score_predictions(rows, [])

    def test_notebook_code_and_embedded_sources_match(self):
        notebook = json.loads((ROOT / 'notebooks/kaggle_en_nn.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        ids = set()
        for index, cell in enumerate(notebook['cells']):
            self.assertNotIn(cell['id'], ids)
            ids.add(cell['id'])
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell_{index}', 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        for name, expected in notebook['metadata']['cosense']['embedded_sha256'].items():
            source = (ROOT / 'scripts' / name).read_text(encoding='utf-8')
            import hashlib
            self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
