"""Pairwise-complete evaluation; undefined correlations are None, never zero."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .data import SCHEMAS


def correlation(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float); valid=np.isfinite(a)&np.isfinite(b)
    a=a[valid]; b=b[valid]
    reason=None
    if len(a)<2: reason='fewer_than_two'
    elif np.ptp(a)==0 or np.ptp(b)==0: reason='constant'
    return dict(value=None if reason else float(spearmanr(a,b).statistic),n=len(a),reason=reason)


def evaluate(frame, task):
    results={}; statistics=[]
    for output in SCHEMAS[task]['labels']:
        y=pd.to_numeric(frame['gold_'+output],errors='coerce'); p=pd.to_numeric(frame['pred_'+output],errors='coerce')
        valid=np.isfinite(y)&np.isfinite(p); f=frame.loc[valid].copy()
        f['_y']=y[valid]; f['_p']=p[valid]
        target_stats=[]; deltas=[]; pairs=0
        for target,g in f.groupby('target',sort=True):
            if len(g)<2: continue
            sy=0. if np.ptp(g['_y'])==0 else float(np.std(g['_y'],ddof=0))
            sp=0. if np.ptp(g['_p'])==0 else float(np.std(g['_p'],ddof=0))
            target_stats.append(dict(target=target,output=output,n=len(g),gold_sd=sy,pred_sd=sp))
            values=g[['_y','_p']].to_numpy()
            errors=[abs((a[0]-b[0])-(a[1]-b[1])) for a,b in combinations(values,2)]
            pairs+=len(errors); deltas.append(float(np.mean(errors)))
        statistics.extend(target_stats)
        results[output]=dict(spearman=correlation(y,p),rmse=float(np.sqrt(np.mean((y[valid]-p[valid])**2))) if valid.any() else None,
                             n_valid=int(valid.sum()),n_total=len(frame),
                             context_variance=correlation([s['gold_sd'] for s in target_stats],[s['pred_sd'] for s in target_stats]),
                             delta_mae_macro=float(np.mean(deltas)) if deltas else None,
                             delta_targets=len(deltas),delta_pairs=pairs)
    vals=[v['spearman']['value'] for v in results.values() if v['spearman']['value'] is not None]
    return dict(outputs=results,mean_valid_output_spearman=float(np.mean(vals)) if vals else None,
                valid_spearman_outputs=len(vals),policy=dict(ddof=0,ties='average ranks',missing='pairwise finite',
                singletons='excluded from SD and deltas',undefined='null with reason and coverage')),pd.DataFrame(statistics)


def compare_predictions(b1,b2,task,replicates=1000,seed=42):
    """Paired target-cluster bootstrap of macro target MAE gain (positive = B2 better)."""
    keys=['task','context_id','target']
    for col in ['seed','split','task']:
        if set(b1[col].unique())!=set(b2[col].unique()) or b1[col].nunique()!=1:
            raise ValueError('Paired identity mismatch: '+col)
    if set(b1['model'])!={'B1'} or set(b2['model'])!={'B2'}: raise ValueError('Expected B1 then B2')
    if b1.duplicated(keys).any() or b2.duplicated(keys).any(): raise ValueError('Duplicate prediction IDs')
    if set(map(tuple,b1[keys].to_numpy()))!=set(map(tuple,b2[keys].to_numpy())): raise ValueError('Prediction coverage differs')
    f=b1.merge(b2,on=keys,suffixes=('_b1','_b2'),validate='one_to_one'); out={}; rng=np.random.default_rng(seed)
    for o in SCHEMAS[task]['labels']:
        a=f['gold_'+o+'_b1'].to_numpy(float); b=f['gold_'+o+'_b2'].to_numpy(float)
        if not np.allclose(a,b,equal_nan=True): raise ValueError('Gold mismatch')
        gain=np.abs(a-f['pred_'+o+'_b1'])-np.abs(a-f['pred_'+o+'_b2'])
        f['context_gain_'+o]=gain
        means=f.assign(_gain=gain).groupby('target')['_gain'].mean().dropna().to_numpy()
        if len(means):
            boot=[float(rng.choice(means,len(means),replace=True).mean()) for _ in range(replicates)]
            out[o]=dict(macro_mae_gain=float(means.mean()),ci95=list(map(float,np.quantile(boot,[.025,.975]))),targets=len(means),replicates=replicates)
        else: out[o]=dict(macro_mae_gain=None,ci95=None,targets=0)
    return out,f


def error_groups(frame,task):
    """Descriptive RQ3 associations, no inferred semantic metadata."""
    f=frame.copy(); group_columns=['alignment_status']
    if 'fallback_reason' in f:
        f['used_fallback']=f['fallback_reason'].fillna('').astype(str).str.len()>0
        group_columns.append('used_fallback')
    if 'truncated' in f: group_columns.append('truncated')
    for col in ['context_length','subword_count']:
        if col in f:
            f[col+'_bin']=pd.qcut(f[col],q=4,duplicates='drop').astype(str)
            group_columns.append(col+'_bin')
    rows=[]
    for col in group_columns:
        for value,g in f.groupby(col,dropna=False):
            for o in SCHEMAS[task]['labels']:
                error=(g['gold_'+o]-g['pred_'+o]).abs()
                rows.append(dict(group=col,value=str(value),output=o,rows=len(g),valid=int(error.notna().sum()),
                                 targets=g['target'].nunique(),mae=float(error.mean()) if error.notna().any() else None))
    return pd.DataFrame(rows)
