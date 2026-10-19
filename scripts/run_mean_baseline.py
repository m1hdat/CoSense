"""Train-only mean predictor; evaluates validation only, never final."""
import argparse
import sys
from data_utils import *
from create_splits import ensure_audit
from validate_splits import validate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config'); a=p.parse_args(); cfg=config(a.config); ensure_audit(cfg)
    reports={s:validate(s,cfg) for s in ['seen_context','unseen_compound']}
    if not all(r['passed'] for r in reports.values()):raise ValueError('Split checks failed; B0 blocked')
    rows,_,_=load('en_nn',cfg); lookup={r['row_key']:r for r in rows}
    for setting in reports:
        train=[lookup[r['row_key']] for r in read_members(setting,'train')]
        val=[lookup[r['row_key']] for r in read_members(setting,'validation')]
        means=[st.mean(float(r[c]) for r in train) for c in ['ModAvg','HeadAvg']]
        predictions=[means[:] for r in val]; scores=metric(val,predictions)
        out=ROOT/'results/b0'/setting
        write_json(out/'metrics.json',{'setting':setting,'split':'validation','train_n':len(train),
          'train_means':dict(zip(['ModAvg','HeadAvg'],means)),'metrics':scores,'final_evaluated':False,
          'split_manifest_sha256':digest(split_dir(setting)/'manifest.json'),'config':cfg})
        records=[{'ContextID':r['ContextID'],'row_key':r['row_key'],'Compound':r['Compound'],
          'pred_ModAvg':means[0],'pred_HeadAvg':means[1],'gold_ModAvg':r['ModAvg'],'gold_HeadAvg':r['HeadAvg']} for r in val]
        write_csv(out/'validation_predictions.tsv',records,list(records[0]),delimiter='\t')
        print(setting,json.dumps({'train_means':means,'validation_metrics':scores},ensure_ascii=True))
    return 0

if __name__=='__main__':sys.exit(main())
