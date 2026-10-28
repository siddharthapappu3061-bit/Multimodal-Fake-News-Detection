"""Offline combined-policy tests plus unchanged six-model CPU/UI regressions."""
import sys,os,json,copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from io import BytesIO
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import pandas as pd
from PIL import Image
from reliability import accepted_mask,selective_metrics,displayed_decision,load_policy,policy_description
from integration_support import read_json
from experiments.reliability_combined.analyze import OUT,GATED,read_validation,partition,tune
from streamlit.testing.v1 import AppTest


class CombinedTests(unittest.TestCase):
    def test_combined_conjunction_and_boundaries(self):
        fake=np.array([.95,.95,.60,.05]);d=np.array([.39,.4,.2,.9])
        threshold={'confidence':.9,'disagreement':.4}
        np.testing.assert_array_equal(accepted_mask(fake,d,'combined',threshold),[True,False,False,False])
        self.assertEqual(accepted_mask([.9],[.399],'combined',threshold).tolist(),[True])
        for bad in [None,.5,{}, {'confidence':.9}, {'confidence':.9,'disagreement':float('nan')}]:
            with self.assertRaises(ValueError): accepted_mask(fake,d,'combined',bad)
        self.assertIn('and disagreement',policy_description({'strategy':'combined','threshold':threshold}))

    def test_metric_denominators_and_display_only(self):
        m=selective_metrics([0,0,1,1],[.99,.65,.01,.8],[.1,.2,.1,.8],'combined',{'confidence':.9,'disagreement':.4})
        self.assertEqual(m['coverage'],.5);self.assertEqual(m['accepted_accuracy'],1)
        self.assertEqual(m['abstained_error_rate'],.5)
        self.assertEqual(m['accepted_fake_recall'],1);self.assertEqual(m['overall_fake_recall'],.5)
        scores={'text':[.9,.1],'clip_image':[.1,.9],'gated_fusion':[.99,.01]}
        before=copy.deepcopy(scores)
        p={'available':True,'strategy':'combined','threshold':{'confidence':.9,'disagreement':.4}}
        self.assertEqual(displayed_decision(scores,p,True),'Requires Verification')
        self.assertEqual(displayed_decision(scores,p,False),'Fake')
        self.assertEqual(displayed_decision(scores,p,True,'text'),'Fake')
        self.assertEqual(displayed_decision(scores,{**p,'available':False},True),'Fake')
        self.assertEqual(scores,before)

    def test_validation_only_and_frozen_selection(self):
        real_read=pd.read_csv
        def read_no_test(path,*args,**kwargs):
            self.assertNotIn('test',str(path).lower())
            return real_read(path,*args,**kwargs)
        with patch('pandas.read_csv',side_effect=read_no_test):
            frame,_=read_validation();training,audit=partition(frame)
            candidates,family,chosen=tune(training)
        frozen=read_json(OUT/'frozen_selection.json')
        self.assertEqual(chosen,frozen['chosen']);self.assertEqual(family,frozen['family'])
        self.assertEqual(len(candidates),99)
        self.assertEqual(len(set(training.id)&set(audit.id)),0)
        altered=frame.copy();altered.loc[altered.id.isin(audit.id),'label']=1-altered.loc[altered.id.isin(audit.id),'label']
        new_training,_=partition(altered)
        self.assertEqual(tune(new_training)[2],chosen)

    def test_fail_closed_policy_loading(self):
        actual=read_json(OUT/'policy.json')
        self.assertFalse(actual['available'])
        self.assertFalse(load_policy(OUT/'policy.json',GATED)['available'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'policy.json'
            valid={**actual,'available':True,'audit_passed':True,'strategy':'combined','threshold':{'confidence':.9,'disagreement':.4}}
            path.write_text(json.dumps(valid));self.assertTrue(load_policy(path,GATED)['available'])
            for update in [{'audit_passed':False},{'manifest_sha256':'wrong'},{'selection_split':'test'},{'threshold':{}},{'run_id':'wrong'}]:
                path.write_text(json.dumps({**valid,**update}))
                self.assertFalse(load_policy(path,GATED)['available'])

    def test_combined_requires_verification_ui_without_reinference(self):
        from gated_runtime import load_six_predictor
        from gated_integration import gated_signature
        from integration_support import bundle_paths
        from dataset import resolve_image
        base,clip=bundle_paths(ROOT)
        predictor=load_six_predictor(str(base),str(clip),str(GATED),gated_signature(base,clip,GATED))
        refs=pd.read_csv(GATED/'results/validate/predictions.csv',dtype={'id':str})
        sample_id=refs.loc[(refs.gated_fake>.1)&(refs.gated_fake<.9),'id'].iloc[0]
        row=pd.read_csv(ROOT/'data/validate_processed.csv',dtype={'id':str}).set_index('id').loc[sample_id]
        stream=BytesIO()
        with Image.open(resolve_image(ROOT/'data',row.local_image_path)) as image:image.save(stream,format='PNG')
        # Simulated validation approval tests UI plumbing only, never persisted as results.
        simulated={**read_json(OUT/'policy.json'),'available':True,'audit_passed':True,'strategy':'combined','threshold':{'confidence':.95,'disagreement':.4}}
        with patch('streamlit.file_uploader',return_value=stream):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            self.assertTrue(app.checkbox(key='selective_classification').disabled)
            app.text_area(key='headline').set_value(row.clean_title).run();app.button[0].click().run()
            original=copy.deepcopy(app.session_state['six_analysis']['scores'])
            with patch('six_model_app.load_policy',return_value=simulated),patch.object(predictor,'analyze',side_effect=AssertionError('Must reuse all predictions')):
                app.run();app.checkbox(key='selective_classification').set_value(True).run()
                self.assertFalse(app.exception);self.assertFalse(app.error)
                self.assertEqual(app.metric[0].value,'Requires Verification')
                self.assertEqual(app.session_state['six_analysis']['scores'],original)
                app.checkbox(key='selective_classification').set_value(False).run()
                self.assertNotEqual(app.metric[0].value,'Requires Verification')
                self.assertEqual(app.session_state['six_analysis']['scores'],original)
            app.sidebar.radio[0].set_value('Model Comparison').run()
            self.assertFalse(app.exception);self.assertFalse(app.error)


if __name__=='__main__':
    # Import classes, not their script runners: earlier verification reports stay untouched.
    from test_gated_integration import GatedIntegrationTests
    from test_reliability import ReliabilityTests
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in [CombinedTests,GatedIntegrationTests,ReliabilityTests])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                training_performed=False,external_apis_used=False,
                checks=['combined boundaries','display-only abstention','validation-only tuning','frozen winner','stale policy rejection',
                        'all six CPU outputs and repeatability','Colab parity','shared encoder/resource caching',
                        'correct Fake/Real, incorrect, high/low disagreement examples','UI pages/default/optional outcome and no recomputation'])
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
