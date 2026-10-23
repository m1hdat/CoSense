"""Immutable input loading, conservative alignment and auditable duplicates."""
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMAS = {
    'nn': dict(target='Compound', parts=['Mod', 'Head'], labels=['ModAvg', 'HeadAvg'], std=['ModStd', 'HeadStd']),
    'pv': dict(target='ParticleVerb', parts=['Base', 'Particle'], labels=['Avg'], std=['Std']),
}

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def find_file(root, name):
    root = Path(root)
    matches = sorted(root.rglob(name)) if root.is_dir() else ([root] if root.name == name else [])
    if len(matches) != 1:
        raise ValueError(f'Expected one {name} under {root}; found {len(matches)}: {matches}. Set DATA_ROOT to a narrower directory.')
    return matches[0]

# Explicit conservative morphology; unlisted irregular forms require manual review.
IRREGULAR = {'take': ['took','taken'], 'give':['gave','given'], 'go':['went','gone'],
             'get':['got','gotten'], 'come':['came'], 'bring':['brought'], 'break':['broke','broken'],
             'run':['ran'], 'put':['put'], 'set':['set'], 'make':['made'], 'keep':['kept'],
             'hold':['held'], 'stand':['stood'], 'fall':['fell','fallen'], 'throw':['threw','thrown'],
             'leave':['left'], 'turn':['turned'], 'cut':['cut'], 'lay':['laid'], 'lie':['lay','lain'],
             'man':['men'], 'woman':['women'], 'child':['children'], 'foot':['feet'], 'tooth':['teeth']}

def forms(word, inflect=True):
    w = word.lower(); result = {w}
    if not inflect: return result
    result.update(IRREGULAR.get(w, []))
    result.update([w+'s', w+'es', w+'ed', w+'ing'])
    if w.endswith('e'): result.update([w+'d', w[:-1]+'ing'])
    if w.endswith('y') and len(w)>1 and w[-2] not in 'aeiou': result.update([w[:-1]+'ies',w[:-1]+'ied'])
    if len(w)>2 and w[-1] not in 'aeiouwxy' and w[-2] in 'aeiou' and w[-3] not in 'aeiou':
        result.update([w+w[-1]+'ed', w+w[-1]+'ing'])
    return result

def occurrences(text, word, inflect=True):
    # IGNORECASE works on the original text: offsets never refer to a normalized copy.
    options = sorted(forms(word, inflect), key=lambda x:(-len(x),x))
    return [(m.start(),m.end()) for m in re.finditer(r'(?<!\w)(?:'+ '|'.join(map(re.escape,options))+r')(?!\w)',text,re.I)]

def align(record):
    text = record['context']; task = record['task']
    a = occurrences(text, record['part1'])
    b = occurrences(text, record['part2'], inflect=task=='nn')
    pairs = []
    for x in a:
        for y in b:
            if x[1] > y[0]: continue
            gap = text[x[1]:y[0]]
            if task=='nn' and re.search(r'[^\s\-–—]',gap): continue
            pairs.append([list(x),list(y)])
    if len(pairs) != 1:
        return dict(spans=None, alignment_status='ambiguous' if pairs else 'missing', candidates=pairs)
    spans=pairs[0]
    exact=all(text[s:e].lower()==record[k].lower() for (s,e),k in zip(spans,['part1','part2']))
    return dict(spans=spans, alignment_status='exact' if exact else 'inflection', candidates=pairs)

def load_records(path, task, labeled=True, overrides=None):
    schema=SCHEMAS[task]; records=[]; seen=set(); overrides=overrides or {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader=csv.DictReader(f, delimiter='\t')
        required=['ContextID', schema['target'], *schema['parts'], 'Context']
        if labeled: required += schema['labels']+schema['std']
        missing=set(required)-set(reader.fieldnames or [])
        if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
        for line,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()): raise ValueError(f'TSV field count mismatch at line {line}')
            if any(not row.get(k,'').strip() for k in ['ContextID',schema['target'],*schema['parts'],'Context']):
                raise ValueError(f'Missing identity/text at line {line}')
            cid=row['ContextID']
            if cid in seen: raise ValueError(f'Duplicate ID {cid}')
            seen.add(cid)
            def numbers(cols):
                vals=[]
                for c in cols:
                    s=row.get(c,'').strip()
                    v=float(s) if s else None
                    if v is not None and not math.isfinite(v): raise ValueError(f'Nonfinite {c} at {cid}')
                    vals.append(v)
                return vals
            r=dict(context_id=cid,task=task,target=row[schema['target']],context=row['Context'],
                   part1=row[schema['parts'][0]],part2=row[schema['parts'][1]],
                   labels=numbers(schema['labels']),annotation_std=numbers(schema['std']))
            if any(v is not None and v<0 for v in r['annotation_std']): raise ValueError(f'Negative Std at {cid}')
            r.update(align(r))
            if cid in overrides:
                spans=overrides[cid]
                if len(spans)!=2 or any(len(s)!=2 or not 0<=s[0]<s[1]<=len(r['context']) for s in spans):
                    raise ValueError(f'Invalid manual spans: {cid}')
                if spans[0][1]>spans[1][0]: raise ValueError(f'Overlapping/reversed spans: {cid}')
                r.update(spans=spans,alignment_status='manual')
            records.append(r)
    if not records: raise ValueError('Empty dataset')
    # B1 must be invariant across occurrences of a target.
    parts=defaultdict(set)
    for r in records: parts[r['target']].add((r['part1'].lower(),r['part2'].lower()))
    bad={k:sorted(v) for k,v in parts.items() if len(v)>1}
    if bad: raise ValueError(f'Inconsistent canonical constituents; resolve explicitly: {bad}')
    return records

def normalized_context(text):
    return ' '.join(re.findall(r'\w+',text.casefold()))

def duplicates(records, threshold=.9):
    """Exact and token-set Jaccard near duplicates. Inverted index avoids all-pairs text scans."""
    exact=[]; near=[]; index=defaultdict(list); sets=[]; texts=[]
    for i,r in enumerate(records):
        text=normalized_context(r['context']); tokens=set(text.split()); candidates=set()
        for t in tokens: candidates.update(index[t])
        for j in sorted(candidates):
            if texts[j]==text: exact.append([records[j]['context_id'],r['context_id']])
            elif len(tokens & sets[j])/max(1,len(tokens | sets[j])) >= threshold:
                near.append([records[j]['context_id'],r['context_id']])
        for t in tokens: index[t].append(i)
        sets.append(tokens); texts.append(text)
    return dict(exact=exact,near=near,near_policy=f'token-set Jaccard >= {threshold}; candidate review, not semantic equivalence')

def audit(records):
    import numpy as np
    def stats(values):
        v=np.array([x for x in values if x is not None],dtype=float)
        return dict(n=len(v),missing=len(values)-len(v),min=float(v.min()) if len(v) else None,
                    max=float(v.max()) if len(v) else None,mean=float(v.mean()) if len(v) else None,
                    sd=float(v.std()) if len(v) else None)
    counts=Counter(r['target'] for r in records)
    return dict(rows=len(records),targets=len(counts),contexts_per_target=dict(sorted(counts.items())),
                alignment=dict(Counter(r['alignment_status'] for r in records)),
                labels={k:stats([r['labels'][i] for r in records]) for i,k in enumerate(SCHEMAS[records[0]['task']]['labels'])},
                annotation_std={k:stats([r['annotation_std'][i] for r in records]) for i,k in enumerate(SCHEMAS[records[0]['task']]['std'])},
                label_scale='Official endpoints/direction unverified; observed distribution only; no clipping')
