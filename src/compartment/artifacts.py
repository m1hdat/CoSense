"""Results bundle and ordered submission contracts."""
import csv
import io
import json
import math
from pathlib import Path
import zipfile
import pandas as pd
from .data import SCHEMAS,dump
from .metrics import evaluate,error_groups


def prediction_frame(records,predictions,model,seed,split,logs=None):
    rows=[]
    for i,(r,p) in enumerate(zip(records,predictions,strict=True)):
        row={k:r[k] for k in ['context_id','task','target','context','part1','part2','alignment_status']}
        row.update(spans=json.dumps(r['spans']),model=model,seed=seed,split=split,context_length=len(r['context'].split()))
        for j,o in enumerate(SCHEMAS[r['task']]['labels']):
            row['gold_'+o]=r['labels'][j]; row['pred_'+o]=float(p[j]); row['annotation_std_'+o]=r['annotation_std'][j]
            row['absolute_error_'+o]=abs(p[j]-r['labels'][j]) if r['labels'][j] is not None else None
        if logs: row.update(logs[i])
        rows.append(row)
    return pd.DataFrame(rows)


def export_bundle(directory,frame,metadata,manifest,history=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    metrics,stats=evaluate(frame,metadata['task'])
    dump(directory/'run_metadata.json',dict(metadata,contract_version=1,output_order=SCHEMAS[metadata['task']]['labels']))
    dump(directory/'metrics.json',metrics); dump(directory/'split_manifest.json',manifest)
    frame.to_csv(directory/'predictions.tsv',sep='\t',index=False)
    frame.to_csv(directory/'error_analysis.tsv',sep='\t',index=False)
    stats.to_csv(directory/'target_statistics.tsv',sep='\t',index=False)
    error_groups(frame,metadata['task']).to_csv(directory/'error_groups.tsv',sep='\t',index=False)
    pd.DataFrame(history or [],columns=['epoch','train_loss','dev_spearman','dev_rmse','seconds']).to_csv(directory/'training_history.csv',index=False)
    names=['run_metadata.json','metrics.json','predictions.tsv','target_statistics.tsv','error_analysis.tsv','training_history.csv','split_manifest.json','error_groups.tsv']
    with zipfile.ZipFile(directory/'results_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names: z.write(directory/name,name)
    return metrics


def read_bundle(source):
    with zipfile.ZipFile(source) as z:
        needed=['run_metadata.json','metrics.json','predictions.tsv','error_analysis.tsv']
        if any(n not in z.namelist() for n in needed): raise ValueError('Incomplete results bundle')
        if sum(i.file_size for i in z.infolist())>200_000_000: raise ValueError('Bundle exceeds 200 MB uncompressed')
        meta=json.loads(z.read('run_metadata.json')); metrics=json.loads(z.read('metrics.json'))
        frame=pd.read_csv(io.BytesIO(z.read('predictions.tsv')),sep='\t',dtype={'context_id':str})
        if meta.get('output_order')!=SCHEMAS[meta['task']]['labels']: raise ValueError('Output order metadata mismatch')
        if frame['context_id'].duplicated().any(): raise ValueError('Duplicate IDs in bundle')
        return meta,metrics,frame


def write_submission(path,records,frame,metadata):
    task=records[0]['task']; labels=SCHEMAS[task]['labels']
    if metadata.get('task')!=task or metadata.get('output_order')!=labels: raise ValueError('Task/output order metadata mismatch')
    ids=[r['context_id'] for r in records]; frame=frame.copy(); frame['context_id']=frame['context_id'].astype(str)
    if frame['context_id'].duplicated().any() or set(ids)!=set(frame['context_id']): raise ValueError('Submission ID coverage mismatch')
    ordered=frame.set_index('context_id').loc[ids]
    values=ordered[['pred_'+o for o in labels]].to_numpy(float)
    if not all(math.isfinite(v) for row in values for v in row): raise ValueError('Nonfinite predictions')
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n')
        for cid,row in zip(ids,values): w.writerow([cid,*[format(v,'.9g') for v in row]])
    validate_submission(path,ids,task)


def validate_submission(path,expected_ids,task):
    rows=list(csv.reader(open(path,encoding='utf-8'),delimiter='\t'))
    if len(rows)!=len(expected_ids): raise ValueError('Row count mismatch')
    if [r[0] for r in rows]!=list(expected_ids) or len(set(expected_ids))!=len(expected_ids): raise ValueError('ID order/coverage mismatch')
    for r in rows:
        if len(r)!=1+len(SCHEMAS[task]['labels']): raise ValueError('Column count mismatch')
        if not all(math.isfinite(float(v)) for v in r[1:]): raise ValueError('Nonfinite prediction')
    return True


def submission_zip(path,files):
    if not files: raise ValueError('No settings selected')
    allowed={'en-nn-pred.tsv','en-pv-pred.tsv'}
    if any(Path(f).name not in allowed for f in files): raise ValueError('Unexpected submission filename')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(f,Path(f).name)
