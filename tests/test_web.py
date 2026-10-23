"""Optional Streamlit interaction tests. All result fixtures are synthetic."""
from pathlib import Path
import sys
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from test_pipeline import fixture_records
from compartment.artifacts import prediction_frame,export_bundle
from compartment.splits import make_splits

ROOT=Path(__file__).resolve().parents[1]


def test_streamlit_data_nn_pv_alignment_filters(tmp_path):
    app=AppTest.from_file(str(ROOT/'app/streamlit_app.py'),default_timeout=60).run()
    assert not app.exception
    nn_file=write_synthetic_tsv(tmp_path,'nn')
    app.text_input[0].set_value(str(nn_file)).run()
    app.button[0].click().run(); assert not app.exception and len(app.dataframe)
    app.checkbox[0].check().run(); assert not app.exception
    app.selectbox[0].select('pv').run()
    app.text_input[0].set_value(str(write_synthetic_tsv(tmp_path,'pv'))).run()
    app.button[0].click().run(); assert not app.exception
    app.checkbox[1].check().run(); assert not app.exception


def test_streamlit_bundles_and_paired_results(tmp_path):
    r=fixture_records(); m=make_splits(r); paths=[]
    for variant,delta in [('B1',1.),('B2',0.)]:
        frame=prediction_frame(r,[[v+delta for v in row['labels']] for row in r],variant,42,'dev')
        out=tmp_path/variant
        export_bundle(out,frame,dict(task='nn',model=variant,seed=42),m)
        paths.append(str(out/'results_bundle.zip'))
    # Inject uploads only in the UI test wrapper; app still reads its actual ZIP contract.
    wrapper=f'''
import io
from pathlib import Path
import streamlit as st
class NamedBuffer(io.BytesIO): pass
files=[]
for path in {paths!r}:
    value=NamedBuffer(Path(path).read_bytes()); value.name=path; files.append(value)
st.file_uploader=lambda label,*a,**kw: files if label.startswith('Import') else None
__file__={str(ROOT/'app/streamlit_app.py')!r}
exec(compile(Path(__file__).read_text(encoding='utf-8-sig'),__file__,'exec'))
'''
    app=AppTest.from_string(wrapper,default_timeout=60).run()
    assert not app.exception
    assert len(app.dataframe)>=3


def write_synthetic_tsv(tmp_path,task):
    import csv
    from compartment.data import SCHEMAS
    schema=SCHEMAS[task]; path=tmp_path/f'en-{task}-train.tsv'
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t'); w.writerow(['ContextID',schema['target'],*schema['parts'],*schema['labels'],*schema['std'],'Context'])
        for r in fixture_records(task):
            w.writerow([r['context_id'],r['target'],r['part1'],r['part2'],*r['labels'],*r['annotation_std'],r['context']])
    return path
