"""Local CPU viewer. This module never imports training, torch or transformers."""
import html
import io
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import streamlit as st
from compartment.data import load_records,audit,duplicates,SCHEMAS
from compartment.artifacts import read_bundle
from compartment.metrics import compare_predictions

st.set_page_config(page_title='ComPartMent Explorer',layout='wide')
st.title('ComPartMent: data & results')
data_tab,result_tab=st.tabs(['Data explorer','Results explorer'])

def highlighted(r):
    spans=r.get('spans')
    if isinstance(spans,str): spans=json.loads(spans)
    text=r['context']
    if not spans: return html.escape(text)
    result=[]; pos=0
    for a,b in sorted(spans):
        result.extend([html.escape(text[pos:a]),'<mark>'+html.escape(text[a:b])+'</mark>']); pos=b
    return ''.join(result)+html.escape(text[pos:])

with data_tab:
    task=st.selectbox('Task',['nn','pv'])
    upload=st.file_uploader('Upload train TSV',type=['tsv'])
    local=st.text_input('Hoặc đường dẫn TSV local',f'data/train/en-{task}-train.tsv')
    if st.button('Đọc dữ liệu'):
        try:
            if upload is not None:
                # Parser accepts a path; use a temporary file, removed immediately.
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp)/'input.tsv'; p.write_bytes(upload.getvalue()); records=load_records(p,task)
            else: records=load_records(local,task)
            st.session_state['records']=records
            st.session_state['dups']=duplicates(records)
        except (ValueError,OSError) as e: st.error(str(e))
    records=st.session_state.get('records')
    if records:
        summary=audit(records); st.json({k:v for k,v in summary.items() if k!='contexts_per_target'})
        st.bar_chart(pd.Series(summary['contexts_per_target'],name='contexts'))
        df=pd.DataFrame(records)
        labels=pd.DataFrame(df['labels'].tolist(),columns=SCHEMAS[records[0]['task']]['labels']); st.bar_chart(labels.apply(pd.Series.value_counts).fillna(0).rename_axis('rating'))
        failures=st.checkbox('Chỉ alignment failures'); dup_only=st.checkbox('Chỉ duplicate contexts')
        if failures: df=df[df.alignment_status.isin(['ambiguous','missing'])]
        dups=st.session_state['dups']; st.write('Duplicate pairs',dups)
        if dup_only:
            ids={i for kind in ['exact','near'] for pair in dups[kind] for i in pair}; df=df[df.context_id.isin(ids)]
        target=st.selectbox('Target',sorted(df.target.unique())) if len(df) else None
        if target:
            rows=df[df.target==target]; st.dataframe(rows.drop(columns=['context']))
            for r in rows.to_dict('records'):
                st.caption(r['context_id']+' / '+r['alignment_status']); st.markdown(highlighted(r),unsafe_allow_html=True)
    else: st.info('Chọn TSV để xem schema, distributions và spans.')
with result_tab:
    files=st.file_uploader('Import một hoặc nhiều results_bundle.zip từ Kaggle',type=['zip'],accept_multiple_files=True)
    bundles=[]
    for file in files:
        try: bundles.append(read_bundle(file))
        except (ValueError,KeyError,OSError,zipfile.BadZipFile) as e: st.error(f'{file.name}: {e}')
    for index,(meta,metrics,frame) in enumerate(bundles):
        st.subheader(f"{meta['task']} / {meta['model']} / seed {meta.get('seed')}")
        st.json(metrics)
        target=st.selectbox('Target',sorted(frame.target.unique()),key=f'target{index}')
        subset=frame[frame.target==target]
        st.line_chart(subset.set_index('context_id')[[c for c in frame if c.startswith(('gold_','pred_'))]])
        st.dataframe(subset)
        st.download_button('Download filtered errors',subset.to_csv(sep='\t',index=False).encode('utf-8'),'filtered_errors.tsv',key=f'down{index}')
    for a in bundles:
        for b in bundles:
            if a[0]['model']=='B1' and b[0]['model']=='B2' and a[0]['task']==b[0]['task'] and a[0]['seed']==b[0]['seed']:
                try:
                    comparison,errors=compare_predictions(a[2],b[2],a[0]['task'])
                    st.write('Paired B1/B2 context gain'); st.json(comparison); st.dataframe(errors)
                except ValueError as e: st.error(str(e))
    if not bundles: st.info('Chưa có results bundle. Web không tải model và không train.')
