"""Real-weight CPU and UI verification. No retraining or data writes."""
import gc
import json
import os
import sys
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
import torch
import pandas as pd
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest
from dataset import resolve_image
from integration_support import bundle_paths, read_json, file_sha256, MODEL_NAMES, REPORT_KEYS
from gated_integration import gated_bundle_path,gated_signature,default_model,SIX_MODEL_NAMES
from gated_runtime import load_six_predictor,load_six_reports
BASE,CLIP=bundle_paths(ROOT)
GATED=gated_bundle_path(ROOT)
OBS={'device':'cpu','training_performed':False,'comparison_tolerance':{'atol':.002,'rtol':.002}}


class GatedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.signature=gated_signature(BASE,CLIP,GATED)
        start=time.perf_counter()
        cls.predictor=load_six_predictor(str(BASE),str(CLIP),str(GATED),cls.signature)
        OBS['cold_load_seconds']=time.perf_counter()-start
        cls.frames={s:pd.read_csv(ROOT/'data'/f'{s}_processed.csv',dtype={'id':str}).set_index('id') for s in ['validate','test']}
        cls.references={s:pd.read_csv(GATED/'results'/s/'predictions.csv',dtype={'id':str}).set_index('id') for s in cls.frames}
        cls.original={s:pd.read_csv(CLIP/'results'/f'{s}_predictions.csv',dtype={'id':str}).set_index('id') for s in cls.frames}
        OBS['run_id']=cls.predictor.gated_run_id
        OBS['manifest_sha256']=file_sha256(GATED/'sha256.json')
        OBS['samples']={}

    def sample(self,split,sample_id):
        row=self.frames[split].loc[sample_id]
        with Image.open(resolve_image(ROOT/'data',row.local_image_path)) as image:
            return row.clean_title,image.copy()

    def test_all_six_parity_and_repeatability(self):
        probes=[('validate',sample['id']) for sample in read_json(GATED.parent/'cpu_verification.json')['samples']]+[('test','bdlmf3'),('test','ba8vze')]
        for split,sample_id in probes:
            title,image=self.sample(split,sample_id)
            start=time.perf_counter(); actual=self.predictor.analyze(title,image); elapsed=time.perf_counter()-start
            self.assertEqual(actual,self.predictor.analyze(title,image))
            self.assertEqual(set(actual['scores']),set(SIX_MODEL_NAMES))
            for kind,probs in actual['scores'].items():
                self.assertEqual(len(probs),2)
                self.assertAlmostEqual(sum(probs),1,places=6)
                expected=self.references[split].loc[sample_id,'gated_fake'] if kind=='gated_fusion' else self.original[split].loc[sample_id,REPORT_KEYS[kind]+'_fake_probability']
                np.testing.assert_allclose(probs[0],expected,atol=.002,rtol=.002)
            for key in ['g_text','g_image','disagreement']:
                np.testing.assert_allclose(actual[key],self.references[split].loc[sample_id,key],atol=.002,rtol=.002)
            original=self.predictor.baseline.predict(title,image)
            for kind,probs in original.items():
                self.assertEqual(probs,actual['scores'][kind])
            OBS['samples'][sample_id]={'split':split, 'seconds':elapsed, **actual}

    def test_exported_loader_matches_integrated_loader(self):
        sys.path.insert(0,str(ROOT/'experiments/gated_fusion'))
        from gated_inference import GatedPredictor
        exported=GatedPredictor(GATED)
        for saved in read_json(GATED.parent/'cpu_verification.json')['samples']:
            title,image=self.sample('validate',saved['id'])
            actual=exported.predict(title,image)
            integrated=self.predictor.analyze(title,image)
            for kind,probs in actual['probabilities'].items():
                np.testing.assert_allclose(probs,saved['prediction']['probabilities'][kind],atol=.002,rtol=.002)
                np.testing.assert_allclose(probs,integrated['scores'][kind],atol=1e-6,rtol=1e-6)
            for key in ['g_text','g_image','disagreement']:
                np.testing.assert_allclose(actual[key],saved['prediction'][key],atol=.002,rtol=.002)
        del exported
        gc.collect()

    def test_devices_dimensions_caching_and_encoder_reuse(self):
        p=self.predictor
        again=load_six_predictor(str(BASE),str(CLIP),str(GATED),self.signature)
        self.assertIs(again,p)
        for model in [p.baseline.encoders.text,p.baseline.encoders.image,p.clip.model,p.gated_head,*p.baseline.heads.values(),*p.clip_heads.values()]:
            self.assertFalse(model.training)
            self.assertTrue(all(param.device.type=='cpu' and not param.requires_grad for param in model.parameters()))
        title,image=self.sample('test','ba8vze')
        with patch.object(p.baseline.encoders,'encode_text',wraps=p.baseline.encoders.encode_text) as text, patch.object(p.clip,'encode',wraps=p.clip.encode) as clip:
            p.analyze(title,image)
            self.assertEqual(text.call_count,1)
            self.assertEqual(clip.call_count,1)
        with torch.inference_mode():
            t=p.baseline.encoders.encode_text([title]); v=p.clip.encode([image])
            logits,gt,gi=p.gated_head(t,v)
        self.assertEqual(logits.shape,(1,2))
        torch.testing.assert_close(gt+gi,torch.ones(1))

    def test_reports_labels_ids_and_promotion_fallback(self):
        reports=load_six_reports(str(BASE),str(CLIP),str(GATED),self.signature)
        self.assertTrue(reports['validation_improved'])
        self.assertAlmostEqual(reports['test']['gated']['accuracy'],.864)
        self.assertEqual(len(reports['dashboard']['models']),6)
        for split,df in self.references.items():
            self.assertEqual(set(df.index),set(self.frames[split].index))
            self.assertTrue(np.array_equal(df.label,self.frames[split].loc[df.index,'2_way_label']))
        with patch('gated_integration.Path.is_file',return_value=False):
            self.assertEqual(default_model(GATED,reports),'clip_fusion')

    def test_fargo_warning_preserves_prediction(self):
        saved=read_json(GATED.parent/'fargo_diagnostic.json')
        with Image.open(ROOT/'experiments/gated_fusion/diagnostics/fargo.jpg') as image:
            actual=self.predictor.analyze(saved['headline'],image)
        self.assertTrue(actual['warning'])
        self.assertGreater(actual['disagreement'],actual['threshold'])
        for kind,probs in saved['measured']['probabilities'].items():
            np.testing.assert_allclose(actual['scores'][kind],probs,atol=.002,rtol=.002)
        # User-labelled Fake remains classified Real: a warning never overrides logits.
        self.assertGreater(actual['scores']['gated_fusion'][1],.5)
        OBS['fargo_qualitative_only']=actual

    def test_ui_pages_and_analysis(self):
        app=AppTest.from_file(str(ROOT/'six_model_app.py'),default_timeout=90).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.selectbox(key='primary_model').options),6)
        for page in ['Model Comparison','Dataset Analysis','Model Architecture']:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            if page=='Model Comparison':
                self.assertEqual(len(app.dataframe[0].value),6)
        saved=read_json(GATED.parent/'fargo_diagnostic.json')
        stream=BytesIO((ROOT/'experiments/gated_fusion/diagnostics/fargo.jpg').read_bytes())
        with patch('streamlit.file_uploader',return_value=stream):
            app=AppTest.from_file(str(ROOT/'six_model_app.py'),default_timeout=90).run()
            app.text_area(key='headline').set_value(saved['headline']).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(len(app.metric),10) # primary + six models + D + two gates
            self.assertEqual(set(app.session_state['six_analysis']['scores']),set(SIX_MODEL_NAMES))
            self.assertTrue(any('disagree substantially' in item.value for item in app.info))
            with patch.object(self.predictor,'analyze',side_effect=AssertionError('Selection must reuse results')):
                for kind in SIX_MODEL_NAMES:
                    app.selectbox(key='primary_model').set_value(kind).run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
            app.text_area(key='headline').set_value('changed headline').run()
            self.assertEqual(len(app.metric),0)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GatedIntegrationTests))
    OBS.update(passed=result.wasSuccessful(),tests_run=result.testsRun)
    path=GATED.parent/'windows_integration_verification.json'
    path.write_text(json.dumps(OBS,indent=2),encoding='utf-8')
    if result.wasSuccessful():
        app=AppTest.from_file(str(ROOT/'six_model_app.py'),default_timeout=90).run()
        promoted=not app.exception and not app.error and app.selectbox(key='primary_model').value=='gated_fusion'
        OBS['promotion_ui_verified']=promoted
        OBS['passed']=bool(promoted)
        path.write_text(json.dumps(OBS,indent=2),encoding='utf-8')
    raise SystemExit(not OBS['passed'])
