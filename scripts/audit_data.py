"""Audit English source TSVs and export reproducible diagnostic figures."""
import argparse
import json
import math
import platform
import re
import statistics as st
import sys
from collections import Counter,defaultdict
from data_utils import *

def audit(branch,cfg):
    rows,fields,issues=load(branch,cfg); t=target(branch)
    if not rows:issues.append({'type':'empty_source'})
    if fields!=(NN if branch=='en_nn' else PV):
        summary={'branch':branch,'schema':fields,'records':len(rows),'unique_expressions':0,'near_duplicate_candidates':0,'alignment_counts':{},'fatal_issues':issues}
        write_json(ROOT/f'reports/audit/{branch}_summary.json',summary)
        return rows,summary
    flags=[]
    for r in rows:
        errors=[]
        for c in fields:
            if not r.get(c,'').strip():errors.append('missing:'+c)
        for c in labels(branch)+[c for c in fields if c.endswith('Std')]:
            try:
                v=float(r[c])
                if not math.isfinite(v):errors.append('nonfinite:'+c)
                elif c.endswith('Std') and v<0:errors.append('negative_std:'+c)
                elif c.endswith('Avg') and not 0<=v<=5:errors.append('out_of_range:'+c)
            except (ValueError,KeyError):errors.append('invalid_numeric:'+c)
        a=alignment(r,branch,cfg)
        flags.append({'row_key':r['row_key'],'ContextID':r['ContextID'],'expression':r[t],
          'Context_original':r['Context'],'Context_normalized':norm(r['Context']),
          'match_class':a['match_class'],'alignment_evidence':a['evidence'],
          'candidate_spans':json.dumps(a['candidates']),'selected_span':json.dumps(a['selected_span']),
          'eligible':not errors,'exclusion_reasons':';'.join(errors),'requires_manual_alignment_review':a['selected_span'] is None})
    dup=duplicate_groups(rows,branch,fields)
    near,tested=near_candidates(rows,cfg)
    if dup['id']:issues.append({'type':'duplicate_id','groups':dup['id']})
    if dup['conflicting_labels']:issues.append({'type':'conflicting_same_sample_labels','groups':dup['conflicting_labels']})
    if any(not f['eligible'] for f in flags):issues.append({'type':'invalid_records','count':sum(not f['eligible'] for f in flags)})
    groups=build_groups(rows,near)
    if groups and max(map(len,groups))>cfg['near_duplicate']['large_group_limit']:
        issues.append({'type':'large_context_chain','maximum':max(map(len,groups))})
    counts=Counter(r[t] for r in rows)
    nums={c:stats([float(r[c]) for r in rows if r[c] and _valid(r[c])]) for c in fields if c.endswith(('Avg','Std'))}
    summary={'branch':branch,'schema':fields,'records':len(rows),'unique_ids':len({r['ContextID'] for r in rows}),
      'unique_expressions':len(counts),'unique_components':{c:len({r[c] for r in rows}) for c in (['Mod','Head'] if branch=='en_nn' else ['Base','Particle'])},
      'contexts_per_expression':stats(list(counts.values())),'context_count_distribution':dict(sorted(Counter(counts.values()).items())),
      'sentence_words':stats([len(re.findall(r'\w+',r['Context'])) for r in rows]),
      'sentence_characters':stats([len(r['Context']) for r in rows]),'numeric':nums,
      'missing':{c:sum(not r.get(c,'').strip() for r in rows) for c in fields},
      'duplicates':{k:{'groups':len(v),'extra_records':sum(len(g)-1 for g in v)} for k,v in dup.items()},
      'near_duplicate_candidates':len(near),'near_pairs_char_scored':tested,
      'context_components':{'count':len(groups),'max_records':max(map(len,groups)) if groups else 0},
      'alignment_counts':dict(Counter(f['match_class'] for f in flags)),
      'eligible_count':sum(f['eligible'] for f in flags),'fatal_issues':issues,'has_time_metadata':False}
    if branch=='en_nn' and not issues:
        by=defaultdict(list)
        for r in rows:by[norm(r[t])].append(r)
        gaps=[float(r['ModAvg'])-float(r['HeadAvg']) for r in rows]
        summary['nn_gap']={'signed':stats(gaps),'absolute':stats([abs(v) for v in gaps]),'absolute_ge_2':sum(abs(v)>=2 for v in gaps)}
        summary['variance_decomposition']={}; summary['descriptive_correlations']={}
        compound_rows=[]
        for c in ['ModAvg','HeadAvg']:
            values=[float(r[c]) for r in rows]; mu=st.mean(values)
            within=sum(sum((float(r[c])-st.mean(float(x[c]) for x in rr))**2 for r in rr) for rr in by.values())/len(rows)
            between=sum(len(rr)*(st.mean(float(r[c]) for r in rr)-mu)**2 for rr in by.values())/len(rows)
            summary['variance_decomposition'][c]={'total_population':st.pvariance(values),'within':within,'between':between,'within_fraction':within/(within+between)}
            lengths=[len(re.findall(r'\w+',r['Context'])) for r in rows]
            stds=[float(r['ModStd' if c=='ModAvg' else 'HeadStd']) for r in rows]
            summary['descriptive_correlations'][c]={'length_score_pearson':correlation(lengths,values),'std_score_pearson':correlation(stds,values),'length_std_pearson':correlation(lengths,stds)}
        for compound,rr in sorted(by.items()):
            rec={'Compound':rr[0]['Compound'],'normalized':compound,'n':len(rr)}
            for c in ['ModAvg','HeadAvg']:
                vv=[float(r[c]) for r in rr]; rec[c+'_mean']=st.mean(vv); rec[c+'_range']=max(vv)-min(vv); rec[c+'_std_population']=st.pstdev(vv)
            compound_rows.append(rec)
        write_csv(ROOT/'reports/audit/en_nn_compound_variation.csv',compound_rows,list(compound_rows[0]))
    write_csv(ROOT/f'reports/audit/{branch}_flags.csv',flags,list(flags[0]) if flags else ['row_key'])
    write_csv(ROOT/f'reports/audit/{branch}_eligible.csv',
              [{'row_key':r['row_key'],'ContextID':r['ContextID'],'source_file':r['source_file']} for r,f in zip(rows,flags) if f['eligible']],
              ['row_key','ContextID','source_file'])
    manual=[f for f in flags if f['requires_manual_alignment_review']]
    write_csv(ROOT/f'reports/audit/{branch}_manual_alignment.csv',manual,list(flags[0]) if flags else ['row_key'])
    near_fields=['left','right','left_id','right_id','token_jaccard','char_5gram_jaccard','left_context','right_context','review_status']
    write_csv(ROOT/f'reports/audit/{branch}_near_candidates.csv',near,near_fields)
    write_json(ROOT/f'reports/audit/{branch}_near_candidates.json',near)
    duplicate_rows=[]
    for kind,gg in dup.items():
        for gi,g in enumerate(gg):
            for i in g:duplicate_rows.append({'kind':kind,'group':gi,'ContextID':rows[i]['ContextID'],'expression':rows[i][t],'Context_original':rows[i]['Context'],'labels':json.dumps({c:rows[i][c] for c in labels(branch)})})
    write_csv(ROOT/f'reports/audit/{branch}_duplicates.csv',duplicate_rows,['kind','group','ContextID','expression','Context_original','labels'])
    write_json(ROOT/f'reports/audit/{branch}_summary.json',summary)
    return rows,summary

def _valid(x):
    try:return math.isfinite(float(x))
    except ValueError:return False

def figures(nn,pv):
    # Optional isolated dependency directory; audit/split core is stdlib.
    if (ROOT/'.runtime').exists():sys.path.insert(0,str(ROOT/'.runtime'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    out=ROOT/'reports/figures'; out.mkdir(parents=True,exist_ok=True)
    colors=['#0072B2','#D55E00']; paths=[]
    def finish(fig,name):
        fig.tight_layout(rect=(0,0,1,.92) if name=='nn_context_variation.png' else (0,0,1,1))
        fig.savefig(out/name,dpi=160,bbox_inches='tight'); plt.close(fig); paths.append(name)
    fig,ax=plt.subplots(figsize=(9,4.5)); bins=np.linspace(0,5,21)
    for c,color in zip(['ModAvg','HeadAvg'],colors):ax.hist([float(r[c]) for r in nn],bins=bins,histtype='step',linewidth=2,label=c,color=color)
    ax.set(title=f'EN-NN: phân bố điểm (n={len(nn)} mẫu)',xlabel='Điểm trung bình của người đánh giá (0–5)',ylabel='Số mẫu'); ax.legend(); finish(fig,'nn_scores.png')
    fig,ax=plt.subplots(figsize=(9,4.5)); ax.hist([float(r['Avg']) for r in pv],bins=bins,color=colors[0],edgecolor='white')
    ax.set(title=f'EN-PV: phân bố điểm (n={len(pv)} mẫu)',xlabel='Điểm trung bình của người đánh giá (0–5)',ylabel='Số mẫu'); finish(fig,'pv_scores.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,rr,t,title,color in zip(axs,[nn,pv],['Compound','ParticleVerb'],['EN-NN','EN-PV'],colors):
        counts=Counter(Counter(r[t] for r in rr).values()); xs=sorted(counts)
        ax.bar(xs,[counts[x] for x in xs],color=color); ax.set(title=f'{title} ({sum(counts.values())} cụm)',xlabel='Số context mỗi cụm',ylabel='Số cụm',xticks=range(1,11))
    finish(fig,'context_counts.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for rr,title,color in zip([nn,pv],['EN-NN','EN-PV'],colors):
        axs[0].hist([len(re.findall(r'\w+',r['Context'])) for r in rr],bins=range(0,61,3),histtype='step',linewidth=2,label=f'{title} n={len(rr)}',color=color)
        axs[1].hist([len(r['Context']) for r in rr],bins=range(0,351,20),histtype='step',linewidth=2,label=f'{title} n={len(rr)}',color=color)
    axs[0].set(xlabel='Độ dài câu (token regex, không phải subword)',ylabel='Số mẫu'); axs[1].set(xlabel='Độ dài câu (ký tự Unicode)',ylabel='Số mẫu')
    for ax in axs:ax.legend()
    finish(fig,'sentence_lengths.png')
    fig,ax=plt.subplots(figsize=(6.5,5.5)); hb=ax.hexbin([float(r['ModAvg']) for r in nn],[float(r['HeadAvg']) for r in nn],gridsize=25,extent=(0,5,0,5),mincnt=1,cmap='Blues')
    ax.plot([0,5],[0,5],ls='--',color='#444444',label='ModAvg = HeadAvg'); ax.set(title=f'EN-NN: hai thành phần (n={len(nn)})',xlabel='ModAvg (điểm)',ylabel='HeadAvg (điểm)',xlim=(0,5),ylim=(0,5)); ax.legend(); fig.colorbar(hb,ax=ax,label='Số mẫu trong ô'); finish(fig,'nn_mod_head.png')
    by=defaultdict(list)
    for r in nn:by[r['Compound']].append(r)
    # No selection by gold or errors: first six alphabetically with >=5 records.
    chosen=sorted(c for c,rr in by.items() if len(rr)>=5)[:6]
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,c in zip(axs.flat,chosen):
        rr=sorted(by[c],key=lambda r:r['ContextID']); xs=list(range(1,len(rr)+1))
        for col,color in zip(['ModAvg','HeadAvg'],colors):ax.plot(xs,[float(r[col]) for r in rr],marker='o',label=col,color=color)
        ax.set(title=f'{c} (n={len(rr)})',xlabel='Context theo thứ tự ID (không phải thời gian)',ylabel='Điểm',ylim=(-.1,5.1),xticks=xs)
    handles,labs=axs.flat[0].get_legend_handles_labels(); fig.legend(handles,labs,loc='upper center',ncol=2); fig.tight_layout(rect=(0,0,1,.95)); finish(fig,'nn_context_variation.png')
    write_json(ROOT/'reports/figures/manifest.json',{'matplotlib':matplotlib.__version__,'numpy':np.__version__,'figures':paths,'variation_compounds':chosen,'selection':'first six alphabetically with >=5 contexts; no selection by score'})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config'); parser.add_argument('--no-figures',action='store_true'); args=parser.parse_args()
    cfg=config(args.config); before=source_snapshot(cfg); results={}; summaries={}
    for b in cfg['sources']:
        results[b],summaries[b]=audit(b,cfg)
        print(b,json.dumps({k:summaries[b][k] for k in ['records','unique_expressions','near_duplicate_candidates','alignment_counts','fatal_issues']},ensure_ascii=True))
    id_branches=defaultdict(set); context_branches=defaultdict(set)
    for b,rows in results.items():
        for r in rows:
            id_branches[r.get('ContextID','')].add(b)
            context_branches[norm(r.get('Context',''))].add(b)
    cross={'ids_in_multiple_branches':sorted(k for k,v in id_branches.items() if len(v)>1),
           'normalized_contexts_in_multiple_branches':sum(len(v)>1 for v in context_branches.values())}
    write_json(ROOT/'reports/audit/cross_branch.json',cross)
    after=source_snapshot(cfg)
    if before!=after:raise RuntimeError('Source changed during audit')
    write_json(ROOT/'reports/audit/manifest.json',{'protocol':cfg,'protocol_sha256':digest(Path(args.config) if args.config else ROOT/'configs/data_protocol.json'),'python':platform.python_version(),
        'sources':before,'source_unchanged':True,'other_tsv_files':[str(p.relative_to(ROOT)) for p in sorted((ROOT/'data').rglob('*.tsv')) if str(p.relative_to(ROOT)).replace('\\','/') not in cfg['sources'].values()],
        'fatal_issues':sum(len(s['fatal_issues']) for s in summaries.values()),'eligible_policy':'no filtering for alignment alone; fatal structural/label problems block all splits; raw unchanged'})
    if any(s['fatal_issues'] for s in summaries.values()):return 1
    if not args.no_figures:figures(results['en_nn'],results['en_pv'])
    return 0

if __name__=='__main__':sys.exit(main())
