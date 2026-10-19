"""Integrity checks; synthetic fixtures never alter real source data."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_utils import *
import validate_splits
import audit_data

class PipelineTests(unittest.TestCase):
    def test_quoted_tab_newline_and_quotes(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'input.tsv'; context='A "quoted" account book\twith tab\nand newline.'
            record=dict(zip(NN,['1','account book','account','book','1','0','2','0',context]))
            write_csv(source,[record],NN,delimiter='\t'); cfg=config(); cfg['sources']['en_nn']=str(source)
            rows,fields,issues=load('en_nn',cfg)
            self.assertEqual(fields,NN);self.assertFalse(issues);self.assertEqual(rows[0]['Context'],context)

    def test_wrong_width_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'input.tsv';source.write_text('\t'.join(NN)+'\n1\tshort\n',encoding='utf-8')
            cfg=config();cfg['sources']['en_nn']=str(source);_,_,issues=load('en_nn',cfg)
            self.assertEqual(issues[0]['type'],'width')

    def test_invalid_labels_exclude_and_block(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'input.tsv'
            record=dict(zip(NN,['1','account book','account','book','6','-1','nan','0','An account book.']))
            write_csv(source,[record],NN,delimiter='\t');cfg=config();cfg['sources']['en_nn']=str(source)
            with patch.object(audit_data,'ROOT',Path(temp)):
                _,report=audit_data.audit('en_nn',cfg)
            self.assertEqual(report['eligible_count'],0);self.assertTrue(report['fatal_issues'])
            with (Path(temp)/'reports/audit/en_nn_flags.csv').open(encoding='utf-8',newline='') as file:
                flag=list(csv.DictReader(file))[0]
            self.assertIn('out_of_range:ModAvg',flag['exclusion_reasons'])
            self.assertIn('negative_std:ModStd',flag['exclusion_reasons'])
            self.assertIn('nonfinite:HeadAvg',flag['exclusion_reasons'])

    def test_alignment_word_boundaries_and_ambiguity(self):
        cfg=config();r={'Compound':'cat box','Mod':'cat','Head':'box','Context':'A bobcat box stood here.'}
        self.assertEqual(alignment(r,'en_nn',cfg)['match_class'],'unresolved')
        r['Context']='cat box beside another Cat Box';result=alignment(r,'en_nn',cfg)
        self.assertEqual(result['match_class'],'multiple_candidates');self.assertIsNone(result['selected_span'])
        r={'ParticleVerb':'put down','Base':'put','Particle':'down','Context':'I put it down.'}
        self.assertEqual(alignment(r,'en_pv',cfg)['match_class'],'discontinuous_candidate')

    def test_near_candidates_and_transitive_components(self):
        rows=[{'ContextID':'1','Context':'one two three four five six seven eight nine ten'},
              {'ContextID':'2','Context':'one two three four five six seven eight nine eleven'}]
        pairs,_=near_candidates(rows,config());self.assertEqual(len(pairs),1)
        rows.append({'Context':'another unrelated sentence'})
        self.assertEqual(build_groups(rows,pairs+[{'left':1,'right':2}]),[[0,1,2]])

    def test_hand_computed_metrics_and_constant_correlation(self):
        rows=[{'Compound':'a','ModAvg':'0','HeadAvg':'0'},{'Compound':'a','ModAvg':'2','HeadAvg':'2'},{'Compound':'b','ModAvg':'4','HeadAvg':'4'}]
        result=metric(rows,[[0,0],[0,0],[0,0]])
        self.assertEqual(result['row_macro_two_output_mae'],2);self.assertEqual(result['compound_macro_two_output_mae'],2.5)
        self.assertIsNone(result['outputs']['ModAvg']['pearson']);self.assertIsNone(result['outputs']['HeadAvg']['spearman'])

    def test_membership_tampering_fails(self):
        setting='seen_context';original={s:read_members(setting,s) for s in SPLITS}
        broken=copy.deepcopy(original);broken['validation'][0]=copy.deepcopy(original['train'][0])
        with patch.object(validate_splits,'read_members',side_effect=lambda setting,s:broken[s]):
            report=validate_splits.validate(setting,config(),write=False)
        self.assertFalse(report['passed']);failed={c['assertion'] for c in report['assertions'] if not c['passed']}
        self.assertIn('eligible_coverage',failed);self.assertIn('unique_rows_and_ids',failed)

    def test_cli_nonzero_and_b0_blocked_on_stale_config(self):
        paths=list((ROOT/'results/b0').rglob('*.*'));before={str(p):digest(p) for p in paths}
        with tempfile.TemporaryDirectory() as temp:
            bad=config();bad['split']['seed']=999;path=Path(temp)/'config.json';path.write_text(json.dumps(bad),encoding='utf-8')
            for script in ['validate_splits.py','run_mean_baseline.py']:
                result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--config',str(path)],capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0,script);self.assertIn('Stale audit',result.stdout+result.stderr)
        self.assertEqual(before,{str(p):digest(p) for p in paths})

if __name__=='__main__':unittest.main()
