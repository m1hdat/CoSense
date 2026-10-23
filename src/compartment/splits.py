"""Target-disjoint splits with exact-duplicate connected components."""
from collections import defaultdict
import hashlib
import json
import random
import numpy as np
from .data import duplicates


def fingerprint(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def make_splits(records, seed=42):
    targets=sorted({r['target'] for r in records})
    if len(targets)<7: raise ValueError('At least 7 targets required for a meaningful 70/15/15 split')
    parent={t:t for t in targets}
    def root(t):
        while parent[t]!=t: t=parent[t]
        return t
    byid={r['context_id']:r for r in records}
    dup=duplicates(records)
    # Keep exact duplicate contexts in one partition even if annotated for different targets.
    for a,b in dup['exact']:
        x,y=root(byid[a]['target']),root(byid[b]['target'])
        parent[max(x,y)]=min(x,y)
    groups=defaultdict(list)
    for t in targets: groups[root(t)].append(t)
    if len(groups)<3: raise ValueError('Exact-duplicate components leave fewer than 3 independent groups')
    means={}
    for t in targets:
        values=[v for r in records if r['target']==t for v in r['labels'] if v is not None]
        means[t]=float(np.mean(values)) if values else 0.
    # Sort by target-mean label, then randomize within short strata before weighted assignment.
    ordered=sorted(groups.values(),key=lambda g:(np.mean([means[t] for t in g]),g))
    rng=random.Random(seed); shuffled=[]
    for i in range(0,len(ordered),10):
        block=ordered[i:i+10]; rng.shuffle(block); shuffled.extend(block)
    desired=np.array([.7,.15,.15])*len(targets); counts=np.zeros(3); assigned={}
    for g in shuffled:
        scores=(desired-counts)/np.maximum(desired,1)
        k=int(np.argmax(scores)); counts[k]+=len(g)
        for t in g: assigned[t]=['train','dev','test'][k]
    partitions={s:[r['context_id'] for r in records if assigned[r['target']]==s] for s in ['train','dev','test']}
    if any(not x for x in partitions.values()): raise ValueError('Empty split: insufficient independent components')
    manifest=dict(version=1,seed=seed,task=records[0]['task'],records_sha256=fingerprint(records),
                  partitions=partitions,target_partition=assigned,policy='target-grouped; label-mean strata; exact-context components',
                  duplicate_report=dup)
    validate_splits(records,manifest)
    manifest['overlap']=overlap_report(records,manifest)
    return manifest

def validate_splits(records,manifest):
    if fingerprint(records)!=manifest['records_sha256']: raise ValueError('Manifest does not match normalized records')
    ids=[r['context_id'] for r in records]; flat=[x for v in manifest['partitions'].values() for x in v]
    if len(flat)!=len(set(flat)) or set(flat)!=set(ids): raise ValueError('Split ID coverage/overlap failure')
    if manifest.get('protocol')=='paired_seen_unseen':
        return validate_diagnostic(records,manifest)
    byid={r['context_id']:r for r in records}; seen={}
    for split,items in manifest['partitions'].items():
        for i in items:
            t=byid[i]['target']
            if t in seen and seen[t]!=split: raise ValueError('Target leakage')
            seen[t]=split
    loc={i:s for s,v in manifest['partitions'].items() for i in v}
    for a,b in duplicates(records)['exact']:
        if loc[a]!=loc[b]: raise ValueError('Exact duplicate context leakage')
    return True

def overlap_report(records,manifest):
    loc={i:s for s,v in manifest['partitions'].items() for i in v}; out={}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        sets=[]
        for s in [a,b]: sets.append({r[p].casefold() for r in records if loc[r['context_id']]==s for p in ['part1','part2']})
        out[a+'_'+b]=dict(constituent_overlap=sorted(sets[0]&sets[1]))
    out['near_duplicates_cross_split']=[p for p in manifest['duplicate_report']['near'] if loc[p[0]]!=loc[p[1]]]
    return out


def seen_unseen_design(records, manifest, support_count=1, seed=42):
    """Additional RQ2 diagnostic: same Q, equal row budgets, independent dev targets."""
    validate_splits(records,manifest); rng=random.Random(seed)
    byid={r['context_id']:r for r in records}; groups=defaultdict(list)
    for i in manifest['partitions']['test']: groups[byid[i]['target']].append(i)
    support=[]; query=[]; excluded=[]
    # Do not separate exact/near duplicate contexts between support and query.
    forbidden={frozenset(p) for k in ['exact','near'] for p in manifest['duplicate_report'][k]}
    for t,ids in sorted(groups.items()):
        ids=sorted(ids); rng.shuffle(ids)
        s=ids[:support_count]; q=[i for i in ids[support_count:] if all(frozenset([i,j]) not in forbidden for j in s)]
        if len(ids)>support_count and q: support+=s; query+=q
        else: excluded.append(t)
    controls=list(manifest['partitions']['train']); rng.shuffle(controls)
    if not support or len(controls)<=len(support): raise ValueError('Insufficient support/query/control examples')
    control=controls[:len(support)]; base=controls[len(support):]
    return dict(query=query,support=support,seen_train=base+support,unseen_train=base+control,
                dev=manifest['partitions']['dev'],excluded_targets=excluded,support_count=support_count,
                confounds=['Control lexical/domain/label distribution may differ','Task-unseen is not pretraining-unseen'],
                source_manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest())


def diagnostic_manifests(records, main_manifest, design):
    """Build explicit paired training pools; no hidden relaxation of main split validation."""
    validate_splits(records,main_manifest)
    pools={}
    for condition in ['seen','unseen']:
        partitions=dict(train=design[condition+'_train'],dev=design['dev'],test=design['query'])
        chosen={i for ids in partitions.values() for i in ids}
        pool=[r for r in records if r['context_id'] in chosen]
        manifest=dict(protocol='paired_seen_unseen',condition=condition,partitions=partitions,
                      records_sha256=fingerprint(pool),task=pool[0]['task'],design=design,
                      seed=main_manifest['seed'])
        validate_splits(pool,manifest); pools[condition]=(pool,manifest)
    if len(pools['seen'][1]['partitions']['train'])!=len(pools['unseen'][1]['partitions']['train']):
        raise ValueError('Unequal paired training budget')
    return pools


def validate_diagnostic(records,manifest):
    byid={r['context_id']:r for r in records}; parts=manifest['partitions']; d=manifest['design']
    expected=dict(train=d[manifest['condition']+'_train'],dev=d['dev'],test=d['query'])
    if parts!=expected: raise ValueError('Diagnostic partitions differ from design')
    sets={s:{byid[i]['target'] for i in ids} for s,ids in parts.items()}
    if sets['dev'] & (sets['train']|sets['test']): raise ValueError('Dev target leakage')
    if manifest['condition']=='unseen' and sets['train'] & sets['test']: raise ValueError('Unseen target leakage')
    if manifest['condition']=='seen' and not sets['test'] <= sets['train']: raise ValueError('Seen queries lack support')
    loc={i:s for s,v in parts.items() for i in v}
    dup=duplicates(records)
    for kind in ['exact','near']:
        for a,b in dup[kind]:
            if loc[a]!=loc[b]: raise ValueError('Diagnostic duplicate context leakage')
    return True
