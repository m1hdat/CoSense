"""All data fixtures in this file are synthetic logic tests, never experimental results."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from compartment.data import align,load_records
from compartment.splits import make_splits,validate_splits,seen_unseen_design,diagnostic_manifests
from compartment.metrics import evaluate,compare_predictions
from compartment.artifacts import prediction_frame,export_bundle,read_bundle,write_submission,validate_submission


def fixture_records(task='nn',targets=20):
    records=[]
    for t in range(targets):
        for c in range(3):
            a=f'word{t}'; b='book' if task=='nn' else 'off'
            r=dict(context_id=f'{t}-{c}',task=task,target=a+' '+b,part1=a,part2=b,
                   context=f'Example{c} contains {a} {b} today.',labels=[float((t+c)%6)]*(2 if task=='nn' else 1),
                   annotation_std=[99.]*(2 if task=='nn' else 1))
            r.update(align(r)); records.append(r)
    return records


def test_split_deterministic_target_disjoint_and_hash():
    r=fixture_records(); m=make_splits(r)
    assert m==make_splits(r)
    assert validate_splits(r,m)
    sets=[{x['target'] for x in r if x['context_id'] in ids} for ids in m['partitions'].values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    other=copy.deepcopy(r); other[0]['labels'][0]=42
    with pytest.raises(ValueError,match='Manifest'): validate_splits(other,m)


def test_duplicate_component_kept_together():
    r=fixture_records(); r[3]['context']=r[0]['context']; m=make_splits(r)
    where={i:s for s,ids in m['partitions'].items() for i in ids}
    assert where[r[0]['context_id']]==where[r[3]['context_id']]


def test_seen_unseen_equal_size_query_excluded():
    r=fixture_records(); m=make_splits(r); d=seen_unseen_design(r,m)
    assert len(d['seen_train'])==len(d['unseen_train'])
    assert not set(d['query']) & (set(d['seen_train'])|set(d['unseen_train']))
    lookup={x['context_id']:x['target'] for x in r}
    assert {lookup[i] for i in d['query']} <= {lookup[i] for i in d['seen_train']}
    assert not {lookup[i] for i in d['query']} & {lookup[i] for i in d['unseen_train']}
    for pool,pm in diagnostic_manifests(r,m,d).values(): assert validate_splits(pool,pm)


def test_alignment_inflection_discontinuous_ambiguity():
    r=dict(task='pv',part1='take',part2='off',context='She took her coat off yesterday.')
    a=align(r); assert a['alignment_status']=='inflection'
    assert [r['context'][s:e] for s,e in a['spans']]==['took','off']
    r['context']='They take off, then take off again.'
    assert align(r)['alignment_status']=='ambiguous'
    r.update(task='nn',part1='account',part2='book',context='ACCOUNT books are here.')
    assert align(r)['alignment_status']=='inflection'








def test_variance_uses_gold_across_contexts_and_deltas():
    r=fixture_records('pv',3)
    for i,x in enumerate(r): x['labels']=[float((i%3)*(i//3+1))]
    pred=np.array([x['labels'] for x in r])*2
    f=prediction_frame(r,pred,'B2',42,'dev'); metrics,stats=evaluate(f,'pv')
    assert metrics['outputs']['Avg']['context_variance']['value']==pytest.approx(1.)
    assert stats['gold_sd'].nunique()==3
    # delta error for [0,1,2] doubled: (1+2+1)/3, macro over multipliers 1,2,3.
    assert metrics['outputs']['Avg']['delta_mae_macro']==pytest.approx(8/3)
    const=f.copy(); const['pred_Avg']=1.
    m,_=evaluate(const,'pv')
    assert m['outputs']['Avg']['spearman']['value'] is None
    assert m['outputs']['Avg']['context_variance']['value'] is None
    assert m['outputs']['Avg']['context_variance']['reason']=='constant'
    f.loc[0,'pred_Avg']=np.nan; m,_=evaluate(f,'pv'); assert m['outputs']['Avg']['n_valid']==8


def test_bundle_and_submission_validation(tmp_path):
    r=fixture_records(); pred=np.array([x['labels'] for x in r]); f=prediction_frame(r,pred,'B2',42,'test')
    m=make_splits(r); meta=dict(task='nn',model='B2',seed=42,output_order=['ModAvg','HeadAvg'])
    export_bundle(tmp_path,f,meta,m)
    a,b,c=read_bundle(tmp_path/'results_bundle.zip'); assert len(c)==len(r)
    path=tmp_path/'en-nn-pred.tsv'; write_submission(path,r,f,meta)
    assert validate_submission(path,[x['context_id'] for x in r],'nn')
    with pytest.raises(ValueError,match='order'): write_submission(path,r,f,dict(meta,output_order=['HeadAvg','ModAvg']))
    with pytest.raises(ValueError,match='coverage'): write_submission(path,r,f.iloc[1:],meta)
    f.loc[0,'pred_ModAvg']=np.inf
    with pytest.raises(ValueError,match='Nonfinite'): write_submission(path,r,f,meta)


def test_paired_bootstrap():
    r=fixture_records('pv'); y=np.array([x['labels'] for x in r]); b1=prediction_frame(r,y+1,'B1',42,'dev'); b2=prediction_frame(r,y,'B2',42,'dev')
    stats,frame=compare_predictions(b1,b2,'pv',100)
    assert stats['Avg']['ci95']==[1.,1.]
    with pytest.raises(ValueError,match='coverage'): compare_predictions(b1,b2.iloc[1:],'pv')





def test_schema_missing_labels_test_mode_and_ids(tmp_path):
    p=tmp_path/'pv.tsv'
    p.write_text('ContextID\tParticleVerb\tBase\tParticle\tContext\n001\ttake off\ttake\toff\tThey take off.\n',encoding='utf-8')
    records=load_records(p,'pv',labeled=False)
    assert records[0]['context_id']=='001' and records[0]['labels']==[None]
    with pytest.raises(ValueError,match='Missing columns'): load_records(p,'pv',labeled=True)
    with p.open('a',encoding='utf-8') as f: f.write('001\ttake off\ttake\toff\tThey take off.\n')
    with pytest.raises(ValueError,match='Duplicate ID'): load_records(p,'pv',labeled=False)




def test_submission_pv_no_labels_and_infinite_file(tmp_path):
    r=fixture_records('pv'); f=prediction_frame(r,[[2.]]*len(r),'B2',42,'official_test')
    path=tmp_path/'en-pv-pred.tsv'; write_submission(path,r,f,dict(task='pv',output_order=['Avg']))
    assert len(path.read_text().splitlines()[0].split('\t'))==2
    path.write_text('x\tinf\n')
    with pytest.raises(ValueError,match='Nonfinite'): validate_submission(path,['x'],'pv')


def test_constant_predictions_sd_exact_zero_with_unequal_group_sizes():
    r=fixture_records('pv',4)
    r=[x for i,x in enumerate(r) if i!=0]
    f=prediction_frame(r,[[3.65982967965616]]*len(r),'B0',42,'dev')
    metrics,stats=evaluate(f,'pv')
    assert (stats['pred_sd']==0).all()
    assert metrics['outputs']['Avg']['context_variance']['value'] is None
