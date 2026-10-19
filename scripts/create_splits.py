"""Create label-blind group splits, only after a clean audit."""
import argparse
import platform
from data_utils import *

def ensure_audit(cfg):
    manifest=read_json(ROOT/'reports/audit/manifest.json')
    if manifest['sources']!=source_snapshot(cfg):raise ValueError('Stale audit: source hashes changed')
    if manifest['protocol']!=cfg:raise ValueError('Stale audit: configuration changed')
    if manifest['fatal_issues']:raise ValueError('Fatal audit issues; split is blocked')
    return manifest

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config'); a=p.parse_args(); cfg=config(a.config)
    ensure_audit(cfg); rows,_,issues=load('en_nn',cfg)
    if issues:raise ValueError(issues)
    near=read_json(ROOT/'reports/audit/en_nn_near_candidates.json')
    fields=['ContextID','row_key','Compound','Compound_normalized','group_id','source_file']
    for setting in ['seen_context','unseen_compound']:
        members,details=allocate(rows,near,cfg,setting)
        for s in SPLITS:write_csv(split_dir(setting)/f'{s}.csv',members[s],fields)
        write_json(split_dir(setting)/'manifest.json',{'setting':setting,'config':cfg,'sources':source_snapshot(cfg),'python':platform.python_version(),
          'eligible_rows':len(rows),'details':details,'hashes':{s:digest(split_dir(setting)/f'{s}.csv') for s in SPLITS},
          'near_duplicate_policy':'all candidates grouped conservatively; semantic review not claimed',
          'counts':{s:{'records':len(members[s]),'compounds':len({r['Compound_normalized'] for r in members[s]}),'fraction':len(members[s])/len(rows)} for s in SPLITS}})
        print(setting,{s:len(members[s]) for s in SPLITS},details)

if __name__=='__main__':main()
