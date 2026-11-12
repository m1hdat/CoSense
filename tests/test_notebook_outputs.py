"""Exercise notebook exports offline with tiny random models, never research results."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from test_pipeline import fixture_records

ROOT = Path(__file__).resolve().parents[1]


def notebook_namespace(variant):
    notebook = json.loads((ROOT / f'notebooks/NN_{variant}_kaggle.ipynb').read_text(encoding='utf-8'))
    namespace = {}
    for index in [6, 7, 8]:
        exec(compile(''.join(notebook['cells'][index]['source']), f'{variant}:cell{index}', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('variant', ['B1', 'B2'])
def test_complete_download_restores_inference_and_resume(tmp_path, variant):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ns = notebook_namespace(variant)
        backbone = tmp_path/'tiny-bert'
        backbone.mkdir()
        vocab = {token: index for index, token in enumerate(
            ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'book', 'contains', 'today', '.']
            + [f'word{i}' for i in range(20)] + [f'example{i}' for i in range(3)])}
        backend = Tokenizer(models.WordPiece(vocab=vocab, unk_token='[UNK]'))
        backend.normalizer = normalizers.BertNormalizer(lowercase=True)
        backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        backend.post_processor = processors.TemplateProcessing(
            single='[CLS] $A [SEP]', special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
        tokenizer = transformers.BertTokenizerFast(tokenizer_object=backend)
        tokenizer.save_pretrained(backbone)
        encoder = transformers.BertModel(transformers.BertConfig(
            vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=24, max_position_embeddings=32))
        encoder.save_pretrained(backbone)
        config = dict(backbone=str(backbone), seed=42, lr=2e-5, batch_size=4,
                      accumulation=1, epochs=2, patience=2, max_length=16, layer=-1,
                      smoke=True, smoke_rows=8, allow_cpu=True, fallback='expression')
        records = fixture_records()
        manifest = ns['make_splits'](records)
        run = tmp_path/'work'/f'NN_{variant}_seed42_smoke'
        ns['dump'](run/'data_provenance.json', {'fixture': 'synthetic'})
        # One real optimizer epoch creates actual best + resumable checkpoints.
        ns['train'](records, manifest, config, variant, run, max_epochs_this_invocation=1)
        (run/'test').mkdir()
        (run/'test'/'optional.txt').write_text('optional test artifact', encoding='utf-8')
        (run/'extra.txt').write_text('future run artifact', encoding='utf-8')
        download = tmp_path/'downloads'
        marker = ns['publish_run_artifacts'](run, download, variant, 'nn', 42, smoke=True)
        inventory = ns['verify_published_artifacts'](marker)
        assert len(list(download.iterdir())) == 3
        complete = download/next(i['name'] for i in inventory['files'] if i['role'] == 'complete')
        extracted = tmp_path/'restored'
        with zipfile.ZipFile(complete) as archive:
            expected = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file()}
            assert set(archive.namelist()) == expected | {'CONTENTS.json'}
            assert archive.testzip() is None
            archive.extractall(extracted)
        assert (extracted/'best.pt').read_bytes() == (run/'best.pt').read_bytes()
        assert (extracted/'resume.pt').read_bytes() == (run/'resume.pt').read_bytes()
        # Offline inference loads only tokenizer/config/weights from the download.
        frame, _ = ns['predict_checkpoint'](extracted, records[:3])
        original, _ = ns['predict_checkpoint'](run, records[:3])
        np.testing.assert_allclose(frame[['pred_ModAvg', 'pred_HeadAvg']],
                                   original[['pred_ModAvg', 'pred_HeadAvg']])
        assert np.isfinite(frame[['pred_ModAvg', 'pred_HeadAvg']].to_numpy()).all()
        assert len(ns['read_bundle'](extracted/'dev/results_bundle.zip')[2]) > 0
        # Restore optimizer/scheduler/RNG state from extracted resume.pt and continue.
        resumed = tmp_path/'resumed'
        ns['train'](records, manifest, config, variant, resumed,
                    resume=extracted/'resume.pt', max_epochs_this_invocation=1)
        state = torch.load(resumed/'resume.pt', map_location='cpu', weights_only=False)
        assert state['epoch'] == 1
        assert len(state['history']) == 2
        # A same-size change must fail SHA256 verification, not just a size check.
        with complete.open('r+b') as target:
            target.seek(100)
            value = target.read(1)
            target.seek(100)
            target.write(bytes([value[0] ^ 1]))
        with pytest.raises(ValueError, match='checksum'):
            ns['verify_published_artifacts'](marker)
        # Never label a results-only download as a complete checkpoint backup.
        (run/'best.pt').unlink()
        with pytest.raises(FileNotFoundError, match='best.pt'):
            ns['publish_run_artifacts'](run, tmp_path/'incomplete', variant, 'nn', 42, smoke=True)
        assert not (tmp_path/'incomplete').exists()
    finally:
        torch.set_num_threads(previous_threads)
