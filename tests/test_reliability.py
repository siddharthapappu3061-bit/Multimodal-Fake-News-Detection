"""Selective metrics and real six-model regressions; never train or change policy."""
import sys,os,json,copy,unittest
from pathlib import Path
from io import BytesIO
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest
from reliability import assess,accepted_mask,selective_metrics,displayed_decision,load_policy
from gated_runtime import load_six_predictor
from gated_integration import gated_signature,gated_bundle_path,SIX_MODEL_NAMES
from integration_support import bundle_paths,read_json,REPORT_KEYS
from dataset import resolve_image
OUT=ROOT/'experiments/reliability'
OBS={}


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.clip=bundle_paths(ROOT);cls.gated=gated_bundle_path(ROOT)
        cls.predictor=load_six_predictor(str(cls.base),str(cls.clip),str(cls.gated),gated_signature(cls.base,cls.clip,cls.gated))
        cls.policy=load_policy(OUT/'policy.json',cls.gated)
        cls.data=pd.read_csv(ROOT/'data/test_processed.csv',dtype={'id':str}).set_index('id')
        cls.reference=pd.read_csv(cls.gated/'results/test/predictions.csv',dtype={'id':str}).set_index('id')
        cls.old=pd.read_csv(cls.clip/'results/test_predictions.csv',dtype={'id':str}).set_index('id')

    def sample(self,sample_id):
        row=self.data.loc[sample_id]
        with Image.open(resolve_image(ROOT/'data',row.local_image_path)) as image:
            return row.clean_title,image.copy()

    def test_metric_denominators_and_empty_groups(self):
        # Hand-calculated fixtures: accepted Fake recall differs from overall recall.
        m=selective_metrics([0,0,1,1],[.99,.6,.1,.55],[.1,.2,.3,.4],'confidence',.9)
        self.assertEqual(m['coverage'],.5)
        self.assertEqual(m['accepted_accuracy'],1)
        self.assertEqual(m['abstained_error_rate'],.5)
        self.assertEqual(m['accepted_fake_recall'],1)
        self.assertEqual(m['overall_fake_recall'],.5)
        none=selective_metrics([0,1],[.5,.5],[0,0],'confidence',.9)
        self.assertIsNone(none['accepted_accuracy'])
        self.assertIsNone(none['accepted_fake_recall'])
        all_=selective_metrics([0,1],[.8,.2],[0,0])
        self.assertIsNone(all_['abstained_error_rate'])
        with self.assertRaises(ValueError): assess(float('nan'),.2,.3)
        with self.assertRaises(ValueError): accepted_mask([.2],[.1],'unknown',.5)

    def test_boundaries_and_only_display_changes(self):
        scores={'text':[.8,.2],'clip_image':[.2,.8],'gated_fusion':[.6,.4]}
        before=copy.deepcopy(scores)
        policy={'available':True,'strategy':'confidence','threshold':.9}
        self.assertEqual(displayed_decision(scores,policy,True),'Requires Verification')
        self.assertEqual(displayed_decision(scores,policy,False),'Fake')
        self.assertEqual(displayed_decision(scores,policy,True,'text'),'Fake')
        self.assertEqual(displayed_decision(scores,{**policy,'available':False},True),'Fake')
        self.assertEqual(scores,before)
        self.assertTrue(accepted_mask([.9],[.4],'confidence',.9)[0])
        self.assertFalse(accepted_mask([.9],[.4],'disagreement',.4)[0])

    def test_frozen_selection_has_no_test_dependency(self):
        from experiments.reliability.analyze import select_policy
        validation=pd.read_csv(OUT/'validation_examples.csv',dtype={'id':str})
        selected,_=select_policy(validation)
        persisted=read_json(OUT/'policy.json')
        for key in selected:
            self.assertEqual(selected[key],persisted[key])
        self.assertEqual(len(set(selected['tune_ids'])&set(selected['audit_ids'])),0)
        self.assertEqual(len(selected['audit_ids']),200)
        self.assertFalse(self.policy['available'])
        self.assertLess(selected['audit_selected']['coverage'],selected['rules']['minimum_coverage'])
        self.assertFalse(load_policy(OUT/'missing_policy.json',self.gated)['available'])

    def test_real_examples_all_six_unchanged(self):
        df=self.reference
        predicted=np.where(df.gated_fake>=.5,0,1)
        correct=predicted==df.label
        cases={'correct_fake':df.index[correct&(df.label==0)][0],
               'correct_real':df.index[correct&(df.label==1)][0],
               'incorrect':df.index[~correct][0],
               'high_disagreement':df.disagreement.idxmax(),
               'low_disagreement':df.disagreement.idxmin()}
        OBS['cases']={}
        for category,sample_id in cases.items():
            title,image=self.sample(sample_id)
            actual=self.predictor.analyze(title,image)
            snapshot=copy.deepcopy(actual)
            decision=displayed_decision(actual['scores'],self.policy,True)
            self.assertEqual(actual,snapshot)
            self.assertEqual(actual,self.predictor.analyze(title,image))
            for model,probs in actual['scores'].items():
                expected=df.loc[sample_id,'gated_fake'] if model=='gated_fusion' else self.old.loc[sample_id,REPORT_KEYS[model]+'_fake_probability']
                np.testing.assert_allclose(probs[0],expected,atol=.002,rtol=.002)
            self.assertNotEqual(decision,'Requires Verification')
            OBS['cases'][category]={'id':sample_id,'label':int(df.loc[sample_id,'label']),'displayed_decision':decision,**actual}

    def test_ui_disabled_policy_and_simulated_approved_policy(self):
        # UI-only simulated approval tests the mechanism, never persisted or reported as a validated policy.
        sample_id=self.reference.index[(self.reference.gated_fake>.1)&(self.reference.gated_fake<.9)][0]
        title,image=self.sample(sample_id)
        stream=BytesIO();image.save(stream,format='PNG')
        with patch('streamlit.file_uploader',return_value=stream):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            self.assertEqual(app.selectbox(key='primary_model').value,'gated_fusion')
            self.assertTrue(app.checkbox(key='selective_classification').disabled)
            app.text_area(key='headline').set_value(title).run();app.button[0].click().run()
            self.assertFalse(app.exception);self.assertFalse(app.error)
            before=copy.deepcopy(app.session_state['six_analysis']['scores'])
            mock_policy={**self.policy,'available':True,'audit_passed':True,'strategy':'confidence','threshold':.95}
            with patch('six_model_app.load_policy',return_value=mock_policy),patch.object(self.predictor,'analyze',side_effect=AssertionError('Display changes cannot run models')):
                app.run()
                app.checkbox(key='selective_classification').set_value(True).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value,'Requires Verification')
                self.assertEqual(app.session_state['six_analysis']['scores'],before)
                app.selectbox(key='primary_model').set_value('text').run()
                self.assertNotEqual(app.metric[0].value,'Requires Verification')
                self.assertEqual(app.session_state['six_analysis']['scores'],before)
            app.sidebar.radio[0].set_value('Model Comparison').run()
            self.assertFalse(app.exception);self.assertFalse(app.error)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReliabilityTests))
    OBS.update(passed=result.wasSuccessful(),tests_run=result.testsRun,training_performed=False,probabilities_modified=False)
    (OUT/'verification.json').write_text(json.dumps(OBS,indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
