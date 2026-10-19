"""EN-NN Kaggle preparation and optional training. Heavy imports stay lazy.

The notebook embeds this file and data_utils.py, so no GitHub access is needed.
Only train/validation are exported or scored. Final membership is integrity-only.
"""
import csv
import importlib.metadata
import json
import math
import platform
import random
import statistics
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from data_utils import NN, SPLITS, allocate, digest, metric, norm, write_csv, write_json

SETTINGS = ('unseen_compound', 'seen_context')
SOURCE = 'data/train/en-nn-train.tsv'
EXPECTED_SOURCE_SHA256 = '98ca6db405a5cdea6cebaaed323ef6e50e3ddd2926b908bbfded8badeee8d895'
BUNDLE_VERSION = 'cosense-en-nn-kaggle-v1'


def payload_paths():
    paths = [SOURCE, 'configs/data_protocol.json', 'reports/audit/en_nn_near_candidates.json']
    for setting in SETTINGS:
        paths.extend(f'data/splits/en_nn/{setting}/{name}'
                     for name in ('train.csv', 'validation.csv', 'final.csv', 'manifest.json'))
    return paths


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter=delimiter, strict=True)
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f'Malformed CSV/TSV: {path}')
        return reader.fieldnames, rows


def verify_bundle(root):
    root = Path(root)
    manifest = read_json(root / 'bundle_manifest.json')
    if manifest['bundle_version'] != BUNDLE_VERSION:
        raise ValueError('Unsupported bundle version; rebuild notebook and dataset together.')
    if set(manifest['files']) != set(payload_paths()):
        raise ValueError('Bundle allowlist differs from expected EN-NN files.')
    for relative, expected in manifest['files'].items():
        path = root / relative
        if not path.is_file() or digest(path) != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError(f'Missing/changed file: {relative}. Re-upload original ZIP without CSV conversion.')
    if digest(root / SOURCE) != EXPECTED_SOURCE_SHA256:
        raise ValueError('Unexpected EN-NN source revision. Do not silently replace raw data.')
    return manifest


def discover_bundle(input_root='/kaggle/input', work_root='/kaggle/working', explicit_root=''):
    """Accept Kaggle-expanded ZIP or an attached ZIP; require one unambiguous bundle."""
    if explicit_root:
        root = Path(explicit_root)
        verify_bundle(root)
        return root
    input_root = Path(input_root)
    expanded = sorted({p.parent for p in input_root.rglob('bundle_manifest.json')})
    if len(expanded) == 1:
        verify_bundle(expanded[0])
        return expanded[0]
    if len(expanded) > 1:
        raise ValueError('Multiple bundles attached. Set DATA_ROOT to one manifest parent directory.')
    archives = sorted(input_root.rglob('cosense-en-nn-data*.zip'))
    if len(archives) != 1:
        raise FileNotFoundError('Attach one cosense-en-nn-data ZIP/Dataset, or set DATA_ROOT. '
                                f'Found {len(archives)} matching ZIPs under {input_root}.')
    archive = archives[0]
    root = Path(work_root) / 'input_cache' / digest(archive)[:16]
    allowed = set(payload_paths()) | {'bundle_manifest.json'}
    with zipfile.ZipFile(archive) as zipped:
        names = [info.filename for info in zipped.infolist()]
        if set(names) != allowed or len(names) != len(allowed):
            raise ValueError('ZIP contains unexpected, duplicate, or missing files.')
        for info in zipped.infolist():
            part = PurePosixPath(info.filename)
            if part.is_absolute() or '..' in part.parts or '\\' in info.filename:
                raise ValueError('Unsafe ZIP member.')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('ZIP symlinks are not supported.')
        if sum(info.file_size for info in zipped.infolist()) > 50_000_000:
            raise ValueError('Unexpectedly large EN-NN bundle.')
        root.mkdir(parents=True, exist_ok=True)
        # Every filename is checked against the fixed relative allowlist first.
        for name in names:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(zipped.read(name))
    verify_bundle(root)
    return root


def load_checked(root, setting):
    if setting not in SETTINGS:
        raise ValueError(f'Unknown setting: {setting}')
    bundle = verify_bundle(root)
    root = Path(root)
    cfg = read_json(root / 'configs/data_protocol.json')
    if cfg['version'] != 'cosense-data-v2' or cfg['sources']['en_nn'] != SOURCE:
        raise ValueError('Unexpected protocol/source path.')
    fields, rows = read_csv(root / SOURCE, '\t')
    if fields != NN or not rows:
        raise ValueError('Unexpected or empty NN schema.')
    for row in rows:
        if any(not value.strip() for value in row.values()):
            raise ValueError('Empty source field.')
        row['row_key'] = f"en_nn:{row['ContextID']}"
        row['source_file'] = SOURCE
        for key in ('ModAvg', 'HeadAvg', 'ModStd', 'HeadStd'):
            value = float(row[key])
            if not math.isfinite(value) or value < 0 or (key.endswith('Avg') and value > 5):
                raise ValueError(f'Invalid source score: {key}')
    lookup = {row['row_key']: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate source ContextID.')
    directory = root / 'data/splits/en_nn' / setting
    split_manifest = read_json(directory / 'manifest.json')
    if (split_manifest['config'] != cfg or split_manifest['setting'] != setting
            or split_manifest['sources']['en_nn']['sha256'] != digest(root / SOURCE)):
        raise ValueError('Stale split manifest/config/source.')
    members = {}
    for split in SPLITS:
        path = directory / f'{split}.csv'
        if digest(path) != split_manifest['hashes'][split]:
            raise ValueError(f'Changed membership: {split}')
        _, members[split] = read_csv(path)
        if not members[split]:
            raise ValueError(f'Empty membership: {split}')
    keys = [member['row_key'] for split in SPLITS for member in members[split]]
    if len(keys) != len(set(keys)) or set(keys) != set(lookup):
        raise ValueError('Split membership duplicate/missing/unrecognized row keys.')
    for split in SPLITS:
        for member in members[split]:
            row = lookup[member['row_key']]
            if (any(member[key] != row[key] for key in ('ContextID', 'Compound', 'source_file'))
                    or member['Compound_normalized'] != norm(row['Compound'])):
                raise ValueError('Membership source fields differ.')
    near = read_json(root / 'reports/audit/en_nn_near_candidates.json')
    replay, _ = allocate(rows, near, cfg, setting)
    if replay != members:
        raise ValueError('Deterministic split replay failed; do not re-split on Kaggle.')
    contexts = {s: {norm(lookup[m['row_key']]['Context']) for m in members[s]} for s in SPLITS}
    compounds = {s: {norm(lookup[m['row_key']]['Compound']) for m in members[s]} for s in SPLITS}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            if contexts[left] & contexts[right]:
                raise ValueError('Context leakage between splits.')
            if setting == 'unseen_compound' and compounds[left] & compounds[right]:
                raise ValueError('Compound leakage in split B.')
    if setting == 'seen_context' and any(not compounds[s] <= compounds['train'] for s in ('validation', 'final')):
        raise ValueError('Split A must have train coverage for evaluation compounds.')
    # Final labels were checked for validity only. No final feature/target export.
    selected = {s: [lookup[m['row_key']] for m in members[s]] for s in ('train', 'validation')}
    report = {'setting': setting, 'source_sha256': digest(root / SOURCE),
              'split_manifest_sha256': digest(directory / 'manifest.json'),
              'bundle_manifest_sha256': digest(root / 'bundle_manifest.json'),
              'bundle_version': bundle['bundle_version'], 'protocol': cfg['version'],
              'counts': {s: len(members[s]) for s in SPLITS},
              'integrity_passed': True, 'final_evaluated': False,
              'final_access': 'source validity and membership integrity only; no final export/prediction/metric'}
    return selected, report


def model_inputs(row, variant):
    if variant == 'B1':
        return {'text': row['Compound']}
    if variant == 'B2':
        return {'text': row['Compound'], 'text_pair': row['Context']}
    raise ValueError('variant must be B1 or B2')


def prepared_record(row, variant):
    return {'metadata': {'row_key': row['row_key'], 'ContextID': row['ContextID']},
            'inputs': model_inputs(row, variant),
            'labels': [float(row['ModAvg']), float(row['HeadAvg'])]}


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def save_prepared(selected, report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'integrity.json', report)
    for variant in ('B1', 'B2'):
        for split, rows in selected.items():
            write_jsonl(output / 'prepared' / variant / f'{split}.jsonl',
                        (prepared_record(row, variant) for row in rows))
    means = [statistics.mean(float(row[key]) for row in selected['train'])
             for key in ('ModAvg', 'HeadAvg')]
    result = {'split': 'validation', 'train_means': means,
              'metrics': metric(selected['validation'], [means[:] for _ in selected['validation']]),
              'final_evaluated': False}
    write_json(output / 'b0_validation.json', result)
    return result


def tokenize_rows(rows, variant, tokenizer, max_length):
    """Fail rather than silently crop an unreviewed target context."""
    records, lengths = [], []
    for row in rows:
        encoding = tokenizer(**model_inputs(row, variant), padding=False, truncation=False,
                             return_token_type_ids=False)
        length = len(encoding['input_ids'])
        lengths.append(length)
        if length > max_length:
            raise ValueError(f"{row['row_key']} {variant} has {length} tokens > {max_length}. "
                             'Review length settings for BOTH models; no automatic cropping.')
        records.append({'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'],
                        'labels': [float(row['ModAvg']), float(row['HeadAvg'])]})
    return records, {'n': len(lengths), 'min': min(lengths), 'max': max(lengths),
                     'mean': statistics.mean(lengths), 'truncated': 0}


def tokenize_all(selected, tokenizer, max_length, output):
    data, report = {}, {}
    for variant in ('B1', 'B2'):
        data[variant], report[variant] = {}, {}
        for split, rows in selected.items():
            records, stats = tokenize_rows(rows, variant, tokenizer, max_length)
            data[variant][split], report[variant][split] = records, stats
            write_jsonl(Path(output) / 'tokens' / variant / f'{split}.jsonl', records)
    write_json(Path(output) / 'token_lengths.json', report)
    return data, report


def environment_report():
    result = {'python': platform.python_version(), 'platform': platform.platform()}
    for package in ('torch', 'transformers', 'tokenizers', 'huggingface-hub', 'safetensors'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def regression_model(model_id=None, revision=None, config_dir=None):
    import torch
    from transformers import AutoConfig, AutoModel

    class TwoScoreRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if config_dir:
                self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(config_dir))
            else:
                self.encoder = AutoModel.from_pretrained(model_id, revision=revision, use_safetensors=True)
            self.head = torch.nn.Linear(self.encoder.config.hidden_size, 2)

        def forward(self, input_ids, attention_mask):
            states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.head(pooled)

    return TwoScoreRegressor()


def make_loader(records, tokenizer, batch_size, shuffle=False, generator=None):
    import torch

    def collate(batch):
        features = [{key: row[key] for key in ('input_ids', 'attention_mask')} for row in batch]
        padded = tokenizer.pad(features, padding=True, return_tensors='pt')
        padded['labels'] = torch.tensor([row['labels'] for row in batch], dtype=torch.float32)
        return padded

    return torch.utils.data.DataLoader(records, batch_size=batch_size, shuffle=shuffle,
                                       collate_fn=collate, num_workers=0, generator=generator)


def predict(model, records, tokenizer, batch_size, device):
    import torch
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in make_loader(records, tokenizer, batch_size):
            inputs = {key: batch[key].to(device) for key in ('input_ids', 'attention_mask')}
            predictions.extend(model(**inputs).cpu().tolist())
    return predictions


def score_predictions(rows, raw):
    if len(raw) != len(rows) or any(len(pair) != 2 or not all(math.isfinite(v) for v in pair) for pair in raw):
        raise ValueError('Invalid/missing/nonfinite predictions.')
    clipped = [[min(5.0, max(0.0, value)) for value in pair] for pair in raw]
    scores = metric(rows, clipped)
    scores['raw_metrics'] = metric(rows, raw)
    scores['out_of_range_fraction'] = sum(value < 0 or value > 5 for pair in raw for value in pair) / (2 * len(raw))
    return scores, clipped


def export_predictions(path, rows, raw):
    scores, clipped = score_predictions(rows, raw)
    records = []
    for row, prediction, clip in zip(rows, raw, clipped):
        records.append({'row_key': row['row_key'], 'ContextID': row['ContextID'], 'Compound': row['Compound'],
                        'raw_ModAvg': prediction[0], 'raw_HeadAvg': prediction[1],
                        'pred_ModAvg': clip[0], 'pred_HeadAvg': clip[1],
                        'gold_ModAvg': row['ModAvg'], 'gold_HeadAvg': row['HeadAvg']})
    write_csv(path, records, list(records[0]), delimiter='\t')
    return scores


def train_variant(selected, tokens, tokenizer, cfg, output, integrity, resume_from=''):
    """Single-GPU FP32 pilot; checkpoints/resume at completed epoch boundaries."""
    import torch
    if cfg['mode'] not in ('smoke', 'train'):
        raise ValueError('Training requires mode smoke or train.')
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. Choose a Kaggle GPU accelerator before training.')
    device = torch.device('cuda:0')
    seed = cfg['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Run directory exists: {output}. Use a new RUN_TAG; no silent overwrite.')
    output.mkdir(parents=True)
    variant = cfg['variant']
    train_tokens = tokens['train']
    if cfg['mode'] == 'smoke':
        train_tokens = train_tokens[:min(128, len(train_tokens))]
    # Validation always remains the entire fixed validation membership.
    model = regression_model(cfg['model_id'], cfg['model_revision']).to(device)
    tokenizer.save_pretrained(output / 'tokenizer')
    model.encoder.config.save_pretrained(output / 'backbone_config')
    write_json(output / 'config.json', cfg)
    write_json(output / 'integrity.json', integrity)
    runtime = environment_report()
    runtime.update({'gpu': torch.cuda.get_device_name(0), 'device_count': torch.cuda.device_count(),
                    'used_gpu_count': 1, 'precision': 'float32', 'torch_cuda': torch.version.cuda,
                    'seed_reproducibility': 'fixed RNG seeds; deterministic algorithms warn_only=True'})
    write_json(output / 'environment.json', runtime)
    write_json(output / 'status.json', {'status': 'running', 'final_evaluated': False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    loader = make_loader(train_tokens, tokenizer, cfg['batch_size'], shuffle=True, generator=generator)
    epochs = 1 if cfg['mode'] == 'smoke' else cfg['epochs']
    steps_per_epoch = len(loader) if cfg['mode'] == 'train' else min(len(loader), cfg['smoke_steps'])
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(total_steps * 0.1))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch, best, stale, global_step = 0, math.inf, 0, 0
    history = []
    compatibility = {'config': cfg, 'integrity': integrity}
    if resume_from:
        # Load ONLY checkpoints produced by this notebook and kept under your control.
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        if checkpoint['compatibility'] != compatibility:
            raise ValueError('Resume config/source/split mismatch. Keep hyperparameters and mode unchanged.')
        if checkpoint['runtime_versions'] != {key: runtime[key] for key in ('torch', 'transformers')}:
            raise ValueError('Resume torch/transformers version mismatch.')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # Optimizer tensors loaded on CPU must follow their parameter device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    # Non-capturable AdamW keeps its scalar step counter on CPU.
                    state[key] = value.cpu() if key == 'step' else value.to(device)
        start_epoch = checkpoint['epoch'] + 1
        best, stale, global_step = checkpoint['best'], checkpoint['stale'], checkpoint['global_step']
        history = checkpoint['history']
        random.setstate(checkpoint['python_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        generator.set_state(checkpoint['loader_rng'])
        source_dir = Path(resume_from).parent
        best_file = source_dir / 'best.safetensors'
        if not best_file.is_file():
            raise FileNotFoundError('Resume needs best.safetensors next to last.pt. Attach the complete run folder.')
        (output / 'best.safetensors').write_bytes(best_file.read_bytes())
        if start_epoch >= epochs or stale >= cfg['patience']:
            raise ValueError('Checkpoint already completed maximum epochs or early stopping; no resume needed.')
    from safetensors.torch import load_file, save_file
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            loss_sum, examples = 0.0, 0
            for step, batch in enumerate(loader):
                if step >= steps_per_epoch:
                    break
                inputs = {key: batch[key].to(device) for key in ('input_ids', 'attention_mask')}
                labels = batch['labels'].to(device)
                optimizer.zero_grad(set_to_none=True)
                predictions = model(**inputs)
                loss = torch.nn.functional.mse_loss(predictions, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss; training stopped.')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                loss_sum += loss.item() * len(labels)
                examples += len(labels)
            raw = predict(model, tokens['validation'], tokenizer, cfg['batch_size'], device)
            scores, _ = score_predictions(selected['validation'], raw)
            value = scores['compound_macro_two_output_mae']
            improved = value < best
            if improved:
                best, stale = value, 0
                save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
                          str(output / 'best.safetensors'))
            else:
                stale += 1
            history.append({'epoch': epoch + 1, 'train_mse': loss_sum / examples,
                            'validation_E': value, 'global_step': global_step,
                            'elapsed_seconds_this_session': time.monotonic() - started})
            write_json(output / 'history.json', history)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'epoch': epoch, 'global_step': global_step,
                          'best': best, 'stale': stale, 'history': history, 'compatibility': compatibility,
                          'runtime_versions': {key: runtime[key] for key in ('torch', 'transformers')},
                          'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
                          'cuda_rng': torch.cuda.get_rng_state_all(), 'loader_rng': generator.get_state()}
            temporary = output / 'last.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(output / 'last.pt')
            print(variant, json.dumps(history[-1]), flush=True)
            if stale >= cfg['patience']:
                break
        model.load_state_dict(load_file(str(output / 'best.safetensors'), device='cpu'))
        raw = predict(model, tokens['validation'], tokenizer, cfg['batch_size'], device)
        scores = export_predictions(output / 'validation_predictions.tsv', selected['validation'], raw)
        result = {'variant': variant, 'setting': cfg['setting'], 'seed': seed, 'mode': cfg['mode'],
                  'split': 'validation', 'train_n_used': len(train_tokens), 'metrics': scores,
                  'best_validation_E': best, 'final_evaluated': False,
                  'elapsed_seconds_this_session': time.monotonic() - started,
                  'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated()}
        write_json(output / 'metrics.json', result)
        write_json(output / 'status.json', {'status': 'completed', 'final_evaluated': False})
        return result
    except BaseException as error:
        write_json(output / 'status.json', {'status': 'failed_or_interrupted', 'error_type': type(error).__name__,
                                          'resume': 'last.pt resumes after the last fully completed epoch only',
                                          'final_evaluated': False})
        raise
    finally:
        del model, optimizer
        torch.cuda.empty_cache()


def load_for_inference(run_dir, device='cpu'):
    """Reload custom mean-pooling + two-output head, rather than AutoModel alone."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    run_dir = Path(run_dir)
    cfg = read_json(run_dir / 'config.json')
    model = regression_model(config_dir=run_dir / 'backbone_config')
    model.load_state_dict(load_file(str(run_dir / 'best.safetensors'), device='cpu'))
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(run_dir / 'tokenizer', local_files_only=True)
    return model, tokenizer, cfg


def predict_texts(run_dir, compounds, contexts=None, device='cpu'):
    model, tokenizer, cfg = load_for_inference(run_dir, device)
    if not compounds or (cfg['variant'] == 'B2' and (contexts is None or len(contexts) != len(compounds))):
        raise ValueError('Provide compounds; B2 needs exactly one context per compound.')
    rows = [{'Compound': compound, 'Context': context, 'ModAvg': '0', 'HeadAvg': '0', 'row_key': f'demo:{i}'}
            for i, (compound, context) in enumerate(zip(compounds, contexts if contexts is not None else [''] * len(compounds)))]
    tokens, _ = tokenize_rows(rows, cfg['variant'], tokenizer, cfg['max_length'])
    raw = predict(model, tokens, tokenizer, cfg['batch_size'], device)
    return [{'raw': pair, 'clipped': [max(0.0, min(5.0, value)) for value in pair]} for pair in raw]
