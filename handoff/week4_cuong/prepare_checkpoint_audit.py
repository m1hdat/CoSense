"""Create B2 Kaggle copies comparing checkpoint selection on one shared training trajectory."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETUP = '''    if resume or final: raise ValueError('Checkpoint audit requires a fresh non-final run')
    audit_best_rmse=float('inf'); audit_best_epoch=None; audit_epochs=[]
    audit_directory=output/'rmse_checkpoint'; audit_directory.mkdir(parents=True,exist_ok=True)
'''
TRACK = '''        audit_epochs.append(dict(epoch=epoch,outputs=copy.deepcopy(metrics['outputs']),
                                 mean_spearman=score,mean_rmse=rmse))
        if rmse<audit_best_rmse:
            audit_best_rmse=rmse; audit_best_epoch=epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch),audit_directory/'best.pt')
'''
EXPORT = '''    # Export the second checkpoint without changing the returned Spearman model.
    audit_metadata=copy.deepcopy(metadata)
    audit_metadata.update(best_epoch=audit_best_epoch,selection_policy='mean_dev_rmse',
                          early_stopping_policy='mean_dev_spearman',shared_training_trajectory=True)
    dump(audit_directory/'run_metadata.json',audit_metadata)
    dump(audit_directory/'config.json',config)
    dump(audit_directory/'split_manifest.json',manifest)
    tokenizer.save_pretrained(audit_directory/'tokenizer')
    model.encoder.config.save_pretrained(audit_directory/'encoder_config')
    model.load_state_dict(torch.load(audit_directory/'best.pt',map_location=device,weights_only=False)['model'])
    audit_frame=prediction_frame(subsets['dev'],predict(model,datasets['dev'],config['batch_size'],device),
                                 variant,config['seed'],'dev',datasets['dev'].logs)
    export_bundle(audit_directory/'dev',audit_frame,audit_metadata,manifest,history)
    dump(output/'epoch_metrics.json',audit_epochs)
    dump(output/'selection_comparison.json',dict(
        spearman_epoch=metadata['best_epoch']+1,rmse_epoch=audit_best_epoch+1,
        min_mean_dev_rmse=audit_best_rmse,early_stopping='unchanged Spearman patience',
        note='Two selectors on the same visited epochs; not a new RMSE early-stopping recipe'))
    model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=False)['model'])
'''


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'Expected one insertion anchor: {before}')
    return text.replace(before, after)


def prepare(task='nn', seed=43):
    notebook = json.loads((ROOT / f'notebooks/{task.upper()}_B2_kaggle.ipynb').read_text())
    training_found = config_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        code = ''.join(cell['source'])
        if 'CONFIG = dict(' in code:
            code = replace_once(code, '\nSEED = 42\n', f'\nSEED = {seed}\n')
            code += "\nCONFIG['checkpoint_audit'] = 'shared_trajectory_rmse_vs_spearman'\n"
            code += "if RESUME is not None or EVALUATE_LOCAL_TEST: raise ValueError('Fresh dev-only audit required')\n"
            config_found = True
        if 'def train(' in code:
            code = replace_once(code, '    started=time.perf_counter()\n', SETUP + '    started=time.perf_counter()\n')
            code = replace_once(code, '        # Undefined rho:', TRACK + '        # Undefined rho:')
            code = replace_once(code, '    return model,tokenizer,metadata\n', EXPORT + '    return model,tokenizer,metadata\n')
            training_found = True
        code = code.replace('/kaggle/working', '/kaggle/working/cuong_week4_selection')
        ast.parse(code)
        cell['source'] = code.splitlines(keepends=True)
        cell['outputs'] = []; cell['execution_count'] = None
    if not training_found or not config_found:
        raise ValueError('Missing config or training cell')
    notebook['cells'].insert(0, dict(cell_type='markdown', metadata={}, source=[
        f'# Cương tuần 4 — {task.upper()} B2 seed{seed}, checkpoint audit\n',
        'Baseline initialization, lr=2e-5. Only checkpoint selection is compared.\n',
        'Early stopping remains Spearman-based. No test evaluation. Run smoke, then full.\n']))
    notebook['cells'].append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source='''# Download both selectors and one complete archive containing both checkpoints.
import shutil
selection_tag=f'{TASK.upper()}_B2_seed{SEED}_{"smoke" if SMOKE else "full"}'
rmse_zip=PUBLISH_ROOT/f'{selection_tag}_RMSE_results.zip'
shutil.copy2(RUN_ROOT/'rmse_checkpoint/dev/results_bundle.zip',rmse_zip)
selection_files=[rmse_zip, PUBLISH_ROOT/f'{selection_tag}_complete.zip',
                 PUBLISH_ROOT/f'{selection_tag}_results.zip']
selection_inventory=dict(task=TASK,model='B2',seed=SEED,smoke=SMOKE,
                         files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in selection_files])
dump(PUBLISH_ROOT/f'{selection_tag}_SELECTION_ARTIFACTS.json',selection_inventory)
print('SELECTION_OUTPUTS_READY:',json.dumps(json.loads((RUN_ROOT/'selection_comparison.json').read_text())))
for p in selection_files:
    print(p)
'''.splitlines(keepends=True)))
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['nn', 'pv'], default='nn')
    parser.add_argument('--seed', type=int, choices=[42,43,44], default=43)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    notebook = prepare(args.task, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(notebook, stream, ensure_ascii=False, indent=1)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
