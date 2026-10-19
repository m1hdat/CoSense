"""Shared, standard-library-only audit/split/evaluation primitives."""
import csv
import hashlib
import json
import math
import random
import re
import statistics as st
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NN = ['ContextID','Compound','Mod','Head','ModAvg','ModStd','HeadAvg','HeadStd','Context']
PV = ['ContextID','ParticleVerb','Base','Particle','Avg','Std','Context']
SPLITS = ('train','validation','final')

def config(path=None):
    return json.loads((Path(path) if path else ROOT/'configs/data_protocol.json').read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def norm(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def write_csv(path, rows, fields, delimiter=','):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter=delimiter,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def load(branch, cfg=None):
    cfg=cfg or config(); path=ROOT/cfg['sources'][branch]
    expected=NN if branch=='en_nn' else PV
    issues=[]; rows=[]
    with path.open(encoding=cfg['parser']['encoding'],newline='') as f:
        reader=csv.DictReader(f,delimiter='\t',quotechar='"',strict=True)
        fields=reader.fieldnames
        if fields!=expected: issues.append({'type':'schema','expected':expected,'actual':fields})
        try:
            for index,r in enumerate(reader,1):
                if None in r or any(v is None for v in r.values()):
                    issues.append({'record':index,'type':'width'}); continue
                r['record_index']=index; r['source_file']=cfg['sources'][branch]
                r['row_key']=f"{branch}:{r.get('ContextID','')}"
                rows.append(r)
        except csv.Error as e:
            issues.append({'type':'csv_parse','line':reader.line_num,'message':str(e)})
    return rows,fields,issues

def labels(branch):
    return ['ModAvg','HeadAvg'] if branch=='en_nn' else ['Avg']

def target(branch):
    return 'Compound' if branch=='en_nn' else 'ParticleVerb'

def quantile(vals,p):
    if not vals:return None
    vals=sorted(vals); pos=(len(vals)-1)*p; lo=int(pos); hi=min(lo+1,len(vals)-1)
    return vals[lo]+(vals[hi]-vals[lo])*(pos-lo)

def stats(vals):
    if not vals:return {'n':0}
    return {'n':len(vals),'min':min(vals),'max':max(vals),'mean':st.mean(vals),'median':st.median(vals),
            'std_population':st.pstdev(vals),'quantiles':{str(p):quantile(vals,p) for p in [0,.05,.25,.5,.75,.95,1]}}

def correlation(x,y):
    if len(x)<2 or st.pstdev(x)==0 or st.pstdev(y)==0:return None
    return st.correlation(x,y)

def ranks(vals):
    order=sorted(range(len(vals)),key=lambda i:vals[i]); out=[0.0]*len(vals); start=0
    while start<len(order):
        end=start+1
        while end<len(order) and vals[order[end]]==vals[order[start]]:end+=1
        for i in order[start:end]:out[i]=(start+end-1)/2+1
        start=end
    return out

class UnionFind:
    def __init__(self,n):self.parent=list(range(n))
    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]]; i=self.parent[i]
        return i
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)
    def groups(self):
        groups=defaultdict(list)
        for i in range(len(self.parent)):groups[self.find(i)].append(i)
        return sorted(groups.values(),key=lambda g:g[0])

def duplicate_groups(rows,branch,fields):
    t=target(branch)
    funcs={'id':lambda r:r['ContextID'], 'full_row':lambda r:tuple(r[c] for c in fields),
           'target_context_raw':lambda r:(r[t],r['Context']),
           'context_raw':lambda r:r['Context'], 'context_normalized':lambda r:norm(r['Context']),
           'target_context_normalized':lambda r:(norm(r[t]),norm(r['Context']))}
    out={}
    for kind,fn in funcs.items():
        buckets=defaultdict(list)
        for i,r in enumerate(rows):buckets[fn(r)].append(i)
        out[kind]=[g for g in buckets.values() if len(g)>1]
    conflicts=[]
    for g in out['target_context_normalized']:
        if len({tuple(r[c] for c in labels(branch)) for r in [rows[i] for i in g]})>1:conflicts.append(g)
    out['conflicting_labels']=conflicts
    out['context_different_target']=[g for g in out['context_normalized'] if len({norm(rows[i][t]) for i in g})>1]
    return out

def near_candidates(rows,cfg):
    """Two independent inverted indexes: complete at configured lexical thresholds.

    This does not detect all paraphrases or semantic equivalence.
    """
    texts=[norm(r['Context']) for r in rows]
    tokens=[set(re.findall(r'\w+',s)) for s in texts]
    grams=[{s[i:i+5] for i in range(max(1,len(s)-4))} for s in texts]
    inverted=defaultdict(list)
    for i,ts in enumerate(tokens):
        for t in ts:inverted[t].append(i)
    settings=cfg['near_duplicate']; shortlisted=set()
    for i,ts in enumerate(tokens):
        hits=Counter(j for t in ts for j in inverted[t] if j>i)
        for j,n in sorted(hits.items()):
            if texts[i]==texts[j]:continue
            tj=n/(len(ts)+len(tokens[j])-n)
            if tj>=settings['token_jaccard']:shortlisted.add((i,j))
    inverted=defaultdict(list)
    for i,gg in enumerate(grams):
        for g in gg:inverted[g].append(i)
    tested=0
    for i,gg in enumerate(grams):
        hits=Counter(j for g in gg for j in inverted[g] if j>i)
        for j,n in hits.items():
            if texts[i]==texts[j]:continue
            tested+=1
            cj=n/(len(gg)+len(grams[j])-n)
            if cj>=settings['char_5gram_jaccard']:shortlisted.add((i,j))
    out=[]
    for i,j in sorted(shortlisted):
        tj=len(tokens[i]&tokens[j])/len(tokens[i]|tokens[j]); cj=len(grams[i]&grams[j])/len(grams[i]|grams[j])
        out.append({'left':i,'right':j,'left_id':rows[i]['ContextID'],'right_id':rows[j]['ContextID'],
                    'token_jaccard':tj,'char_5gram_jaccard':cj,'left_context':rows[i]['Context'],
                    'right_context':rows[j]['Context'],'review_status':'unconfirmed_semantics_conservatively_grouped'})
    return out,tested

# Deliberately small and transparent heuristic lexicon, not an NLP lemmatizer.
IRREGULAR=dict(pair.split(':') for pair in ('be:am,is,are,was,were,been,being have:has,had do:does,did,done go:goes,went,gone '
 'take:took,taken come:came,come get:got,gotten put:put set:set cut:cut let:let hit:hit shut:shut hurt:hurt '
 'run:ran,run sit:sat stand:stood give:gave,given make:made find:found leave:left keep:kept '
 'bring:brought buy:bought think:thought catch:caught teach:taught fight:fought seek:sought '
 'tell:told sell:sold say:said pay:paid lay:laid lie:lay,lain hold:held read:read lead:led '
 'write:wrote,written drive:drove,driven ride:rode,ridden rise:rose,risen fall:fell,fallen '
 'break:broke,broken speak:spoke,spoken throw:threw,thrown blow:blew,blown grow:grew,grown '
 'know:knew,known draw:drew,drawn eat:ate,eaten beat:beat,beaten bite:bit,bitten hide:hid,hidden '
 'shake:shook,shaken steal:stole,stolen wear:wore,worn tear:tore,torn strike:struck,stricken '
 'hang:hung choose:chose,chosen freeze:froze,frozen forget:forgot,forgotten lose:lost '
 'send:sent spend:spent build:built feel:felt meet:met sleep:slept sweep:swept win:won '
 'swim:swam,swum sing:sang,sung ring:rang,rung drink:drank,drunk sink:sank,sunk').split())

def forms(word,verb=False):
    w=norm(word); out={w,w+'s'}
    if w.endswith(('s','x','z','ch','sh')):out.add(w+'es')
    if w.endswith('y') and len(w)>1 and w[-2] not in 'aeiou':out.add(w[:-1]+'ies')
    if verb:
        out.update(IRREGULAR.get(w,'').split(',')); out.update([w+'ed',w+'ing'])
        if w.endswith('e'):out.update([w+'d',w[:-1]+'ing'])
        if w.endswith('y'):out.add(w[:-1]+'ied')
        if len(w)>2 and w[-1] not in 'aeiouwxy' and w[-2] in 'aeiou' and w[-3] not in 'aeiou':
            out.update([w+w[-1]+'ed',w+w[-1]+'ing'])
    else:
        out.update({'man':'men','woman':'women','child':'children','foot':'feet','tooth':'teeth','mouse':'mice'}.get(w,'').split(','))
    return out-{''}

def alignment(r,branch,cfg):
    phrase=r[target(branch)]; context=r['Context']; pat=r'(?<!\w)'+re.escape(phrase)+r'(?!\w)'
    exact=list(re.finditer(pat,context)); insensitive=list(re.finditer(pat,context,re.IGNORECASE))
    matches=insensitive
    if matches:
        return {'match_class':'multiple_candidates' if len(matches)>1 else ('exact_contiguous' if exact else 'case_insensitive_contiguous'),
                'evidence':'surface_with_word_boundaries','candidates':[[[m.start(),m.end()]] for m in matches],
                'selected_span':[[matches[0].start(),matches[0].end()]] if len(matches)==1 else None}
    words=list(re.finditer(r"\w+(?:['’]\w+)?",context)); values=[norm(m.group()) for m in words]
    left=r['Mod' if branch=='en_nn' else 'Base']; right=r['Head' if branch=='en_nn' else 'Particle']
    if len(re.findall(r'\w+',left))!=1 or len(re.findall(r'\w+',right))!=1:
        return {'match_class':'unresolved','evidence':'multiword_component_not_supported','candidates':[],'selected_span':None}
    lf=forms(left,branch=='en_pv'); rf=forms(right,False) if branch=='en_nn' else {norm(right)}
    candidates=[]; adjacent=[]
    for i,w in enumerate(values):
        if w not in lf:continue
        for j,v in enumerate(values):
            if j<=i or v not in rf:continue
            distance=j-i
            if distance==1 or (branch=='en_pv' and distance<=cfg['alignment']['pv_max_token_distance']):
                span=[[words[i].start(),words[i].end()],[words[j].start(),words[j].end()]]
                candidates.append(span)
                if distance==1:adjacent.append(span)
    # Multiple alternatives are retained, never silently choose the first.
    if len(candidates)>1:kind='multiple_candidates'
    elif adjacent:kind='inflection_or_punctuation_candidate'
    elif candidates:kind='discontinuous_candidate'
    else:kind='unresolved'
    return {'match_class':kind,'evidence':'heuristic_not_linguistic_confirmation','candidates':candidates,'selected_span':None}

def build_groups(rows,near,compound=False):
    uf=UnionFind(len(rows)); buckets=defaultdict(list)
    for i,r in enumerate(rows):buckets[norm(r['Context'])].append(i)
    for group in buckets.values():
        for i in group[1:]:uf.union(group[0],i)
    for pair in near:uf.union(pair['left'],pair['right'])
    if compound:
        buckets=defaultdict(list)
        for i,r in enumerate(rows):buckets[norm(r['Compound'])].append(i)
        for group in buckets.values():
            for i in group[1:]:uf.union(group[0],i)
    return uf.groups()

def allocate(rows,near,cfg,setting):
    groups=build_groups(rows,near,compound=setting=='unseen_compound')
    rng=random.Random(cfg['split']['seed']); shuffled=groups.copy(); rng.shuffle(shuffled)
    assignment={}; comp_groups=defaultdict(set)
    for gi,g in enumerate(groups):
        for i in g:comp_groups[norm(rows[i]['Compound'])].add(gi)
    forced=set()
    if setting=='seen_context':
        for memberships in comp_groups.values():
            if len(memberships)<=cfg['split']['small_compound_groups_train_only']:forced.update(memberships)
        # Choose train coverage from seed-shuffled order without using labels.
        covered=set()
        for gi in sorted(forced):
            assignment[gi]='train'; covered.update(norm(rows[i]['Compound']) for i in groups[gi])
        gindex={tuple(g):gi for gi,g in enumerate(groups)}
        for g in shuffled:
            gi=gindex[tuple(g)]; compounds={norm(rows[i]['Compound']) for i in g}
            if not compounds<=covered:
                assignment[gi]='train'; covered.update(compounds)
    counts=Counter()
    for gi,s in assignment.items():counts[s]+=len(groups[gi])
    # Seed-shuffled order avoids routing most large compounds into train.
    # Balance record counts, never use labels or baseline/model scores.
    ordered=shuffled; gindex={tuple(g):gi for gi,g in enumerate(groups)}
    goals={s:len(rows)*cfg['split']['ratios'][s] for s in SPLITS}
    for g in ordered:
        gi=gindex[tuple(g)]
        if gi in assignment:continue
        s=min(SPLITS,key=lambda s:sum((counts[k]+(len(g) if k==s else 0)-goals[k])**2 for k in SPLITS))
        assignment[gi]=s; counts[s]+=len(g)
    members={s:[] for s in SPLITS}
    for gi,g in enumerate(groups):
        for i in g:
            members[assignment[gi]].append({'ContextID':rows[i]['ContextID'],'row_key':rows[i]['row_key'],
              'Compound':rows[i]['Compound'],'Compound_normalized':norm(rows[i]['Compound']),
              'group_id':f'g{gi:05d}','source_file':rows[i]['source_file']})
    for s in SPLITS:members[s].sort(key=lambda r:r['ContextID'])
    return members,{'groups':len(groups),'max_group_records':max(map(len,groups)),
                    'forced_train_only_groups':len(forced),'algorithm':'seed42_shuffled_group_squared_row_deficit_v2'}

def source_snapshot(cfg):
    return {b:{'path':p,'bytes':(ROOT/p).stat().st_size,'sha256':digest(ROOT/p)} for b,p in cfg['sources'].items()}

def split_dir(setting):return ROOT/'data/splits/en_nn'/setting

def read_members(setting,s):
    with (split_dir(setting)/f'{s}.csv').open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def metric(rows,predictions):
    result={'n':len(rows),'compounds':len({norm(r['Compound']) for r in rows}),'outputs':{}}
    row_maes=[]
    for c,idx in [('ModAvg',0),('HeadAvg',1)]:
        y=[float(r[c]) for r in rows]; p=[v[idx] for v in predictions]
        errors=[abs(a-b) for a,b in zip(y,p)]
        result['outputs'][c]={'mae':st.mean(errors),'rmse':math.sqrt(st.mean((a-b)**2 for a,b in zip(y,p))),
          'pearson':correlation(y,p),'spearman':correlation(ranks(y),ranks(p)),
          'correlation_status':'NA: constant predictions' if st.pstdev(p)==0 else 'defined_or_constant_gold'}
        row_maes.append(errors)
    per_row=[(a+b)/2 for a,b in zip(*row_maes)]; by=defaultdict(list)
    for r,e in zip(rows,per_row):by[norm(r['Compound'])].append(e)
    result['row_macro_two_output_mae']=st.mean(per_row)
    result['compound_macro_two_output_mae']=st.mean(st.mean(v) for v in by.values())
    return result
