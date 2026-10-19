"""Independent membership checks; nonzero exit status for any failure."""
import argparse
import sys
from data_utils import *
from create_splits import ensure_audit

def validate(setting,cfg,write=True):
    checks=[]
    def check(name,ok,details=None):checks.append({'assertion':name,'passed':bool(ok),'details':details})
    rows,fields,issues=load('en_nn',cfg); lookup={r['row_key']:r for r in rows}
    near=read_json(ROOT/'reports/audit/en_nn_near_candidates.json')
    saved=read_json(split_dir(setting)/'manifest.json'); members={s:read_members(setting,s) for s in SPLITS}
    check('schema_and_parse',fields==NN and not issues)
    check('source_hashes_current',saved['sources']==source_snapshot(cfg))
    check('config_current',saved['config']==cfg)
    check('membership_file_hashes',all(digest(split_dir(setting)/f'{s}.csv')==saved['hashes'][s] for s in SPLITS))
    combined=[r for mm in members.values() for r in mm]; keys=[r['row_key'] for r in combined]
    check('unique_rows_and_ids',len(keys)==len(set(keys)) and len({r['ContextID'] for r in combined})==len(combined))
    check('eligible_coverage',set(keys)==set(lookup) and len(keys)==len(rows))
    check('source_fields_match',all(r['row_key'] in lookup and all(r[c]==lookup[r['row_key']][c] for c in ['ContextID','Compound','source_file']) and r['Compound_normalized']==norm(lookup[r['row_key']]['Compound']) for r in combined))
    rowsets={s:{r['row_key'] for r in mm} for s,mm in members.items()}
    compoundsets={s:{r['Compound_normalized'] for r in mm} for s,mm in members.items()}
    contextsets={s:{norm(lookup[r['row_key']]['Context']) for r in mm if r['row_key'] in lookup} for s,mm in members.items()}
    for i,s in enumerate(SPLITS):
        for t in SPLITS[i+1:]:
            check(f'no_id_overlap_{s}_{t}',not(rowsets[s]&rowsets[t]))
            check(f'no_normalized_context_overlap_{s}_{t}',not(contextsets[s]&contextsets[t]))
            if setting=='unseen_compound':check(f'no_compound_overlap_{s}_{t}',not(compoundsets[s]&compoundsets[t]))
    if setting=='seen_context':
        check('all_evaluation_compounds_seen_in_train',all(compoundsets[s]<=compoundsets['train'] for s in ['validation','final']))
    placement={r['row_key']:s for s,mm in members.items() for r in mm}
    protected=all(placement.get(rows[p['left']]['row_key'])==placement.get(rows[p['right']]['row_key']) for p in near)
    check('all_near_candidates_kept_together',protected,{'candidate_count':len(near),'semantic_confirmation':False})
    check('valid_labels',all(math.isfinite(float(r[c])) and 0<=float(r[c])<=5 for r in rows for c in ['ModAvg','HeadAvg']))
    check('valid_std',all(math.isfinite(float(r[c])) and float(r[c])>=0 for r in rows for c in ['ModStd','HeadStd']))
    expected,_=allocate(rows,near,cfg,setting)
    check('deterministic_recreation',all(members[s]==expected[s] for s in SPLITS))
    counts={}
    for s,mm in members.items():
        rr=[lookup[r['row_key']] for r in mm if r['row_key'] in lookup]
        counts[s]={'records':len(rr),'compounds':len(compoundsets[s]),'fraction':len(rr)/len(rows),
          'contexts_per_compound':stats(list(Counter(norm(r['Compound']) for r in rr).values())),
          'labels':{c:stats([float(r[c]) for r in rr]) for c in ['ModAvg','HeadAvg']},
          'context_groups':len({r['group_id'] for r in mm}),
          'component_overlap_with_train':{c:len({norm(r[c]) for r in rr}&{norm(lookup[r['row_key']][c]) for r in members['train'] if r['row_key'] in lookup}) for c in ['Mod','Head']}}
    report={'setting':setting,'passed':all(c['passed'] for c in checks),'assertions':checks,'counts':counts,
      'near_duplicate_status':'candidate semantics unconfirmed; every detected candidate protected by same-split conservative grouping; undetected paraphrases not ruled out',
      'final_labels_used_for':'data validity/distribution only, no model predictions or final metrics'}
    if write:write_json(ROOT/f'reports/audit/split_validation_{setting}.json',report)
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config'); a=p.parse_args(); cfg=config(a.config)
    try:
        ensure_audit(cfg)
        reports=[validate(s,cfg) for s in ['seen_context','unseen_compound']]
        for r in reports:print(r['setting'],'PASS' if r['passed'] else 'FAIL',len(r['assertions']),'assertions')
        return 0 if all(r['passed'] for r in reports) else 1
    except (ValueError,KeyError,FileNotFoundError) as e:
        print('FAIL:',e,file=sys.stderr); return 1

if __name__=='__main__':sys.exit(main())
