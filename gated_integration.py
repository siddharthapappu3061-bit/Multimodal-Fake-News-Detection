"""Six-model CPU inference and validated gated report contracts; no training."""
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from integrated_inference import FiveModelPredictor
from integration_support import MODEL_NAMES, read_json, file_sha256, load_evaluation, asset_signature
from experiments.gated_fusion.gated_model import GatedFusion
from experiments.gated_fusion.gated_analysis import metrics, choose_threshold, disagreement_report, warning_policy
from models import ENCODER_CONFIG
from clip_models import FEATURE_SPEC

SIX_MODEL_NAMES = {**MODEL_NAMES, 'gated_fusion': 'DistilBERT + CLIP gated fusion'}


def gated_bundle_path(project):
    return Path(os.environ.get('FAKEDDIT_GATED_BUNDLE', Path(project) / 'experiments/gated_fusion/imported/windows_export/gated_artifacts')).resolve()


def gated_signature(base, clip, gated):
    gated = Path(gated)
    required = ['sha256.json', 'config.json', 'gated_fusion.pt', 'selection.json', 'experiment_config.json',
                'results/validate/metrics.json', 'results/validate/predictions.csv',
                'results/test/metrics.json', 'results/test/predictions.csv']
    missing = [str(gated / name) for name in required if not (gated / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing gated artifacts:\n' + '\n'.join(missing))
    manifest = read_json(gated / 'sha256.json')
    paths = [gated / name for name in dict.fromkeys([*required, *manifest])]
    for path in paths:
        if not path.resolve().is_relative_to(gated) or not path.is_file():
            raise FileNotFoundError(f'Missing or unsafe gated artifact: {path}')
    return asset_signature(base, clip) + tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)


def verify_bundle(gated):
    gated = Path(gated).resolve()
    manifest = read_json(gated / 'sha256.json')
    required = {'config.json', 'gated_fusion.pt', 'selection.json', 'experiment_config.json',
                'models/text.pt', 'models/clip_image.pt', 'models/clip_fusion.pt'}
    required.update(f'results/{s}/{f}' for s in ['validate','test'] for f in ['metrics.json','predictions.csv'])
    required.update('encoders/' + name for name in ['clip/config.json','clip/model.safetensors','clip/preprocessor_config.json',
        'distilbert/config.json','distilbert/model.safetensors','distilbert/tokenizer.json', 'distilbert/tokenizer_config.json',
        'distilbert/vocab.txt','distilbert/special_tokens_map.json'])
    if not required.issubset(manifest):
        raise ValueError('Incomplete gated export manifest: ' + ', '.join(sorted(required-set(manifest))))
    for name, expected in manifest.items():
        path = (gated / name).resolve()
        if not path.is_relative_to(gated) or not path.is_file():
            raise FileNotFoundError(f'Missing or unsafe gated artifact: {name}')
        if file_sha256(path) != expected:
            raise ValueError(f'Gated artifact checksum mismatch: {name}')
    return manifest


def verified_reports(base, clip, gated):
    """Recompute metrics/threshold from saved predictions, checking IDs against original reports."""
    base, clip, gated = map(Path, (base, clip, gated))
    config = read_json(gated / 'config.json')
    baseline_report = load_evaluation(base, clip)
    base_config, clip_config = read_json(base/'config.json'), read_json(clip/'config.json')
    if (config['baseline_run_id'] != base_config['run_id'] or config['clip_run_id'] != clip_config['run_id']
            or config['labels'] != {'0':'Fake','1':'Real'} or config['feature_spec'] != FEATURE_SPEC
            or config['text_encoder'] != ENCODER_CONFIG):
        raise ValueError('Gated export does not match the original model/label/preprocessing contract')
    result = {}
    for split in ('validate', 'test'):
        frame = pd.read_csv(gated / 'results' / split / 'predictions.csv', dtype={'id':str}).set_index('id')
        original = pd.read_csv(clip / 'results' / f'{split}_predictions.csv', dtype={'id':str}).set_index('id')
        if len(frame)!=500 or not frame.index.is_unique or not original.index.is_unique or set(frame.index)!=set(original.index):
            raise ValueError(f'{split}: expected the identical 500 original sample IDs')
        original = original.loc[frame.index]
        if not np.array_equal(frame.label, original.label):
            raise ValueError(f'{split}: labels changed')
        for current, old in [('text_fake','distilbert_fake_probability'),('clip_image_fake','clip_image_fake_probability'),('concatenation_fake','clip_fusion_fake_probability')]:
            if not np.allclose(frame[current],original[old],atol=.002,rtol=.002):
                raise ValueError(f'{split}: original classifier predictions differ: {current}')
        saved = read_json(gated / 'results' / split / 'metrics.json')
        if saved['split'] != split:
            raise ValueError('Report split mismatch')
        for kind,column in [('gated','gated_fake'),('concatenation','concatenation_fake')]:
            actual = metrics(frame.label, frame[column])
            for key,value in actual.items():
                same = value == saved[kind][key] if key=='confusion_matrix' or value is None else math.isclose(value,saved[kind][key],abs_tol=1e-9)
                if not same:
                    raise ValueError(f'{split}/{kind}/{key}: report does not match predictions')
        if split == 'validate':
            contract = read_json(gated / 'experiment_config.json')
            threshold = choose_threshold(frame.text_fake,frame.clip_image_fake,quantile=contract['threshold_quantile'],fixed=contract['fixed_threshold'])
            policy = warning_policy(disagreement_report(frame.label,frame.text_fake,frame.clip_image_fake,frame.gated_fake,threshold))
            if policy['enabled'] != config['warning_policy']['enabled'] or not math.isclose(policy['threshold'],config['warning_policy']['threshold'],abs_tol=1e-7) or config['warning_policy']['changes_label']:
                raise ValueError('Warning policy differs from the validation-selected policy')
        result[split] = saved
    selected = read_json(gated / 'selection.json')
    if config['seed'] != selected['selected_seed']:
        raise ValueError('Selected seed differs from exported configuration')
    m=result['test']['gated']
    dashboard = {**baseline_report, 'models': {**baseline_report['models'], 'gated_fusion': {
        'accuracy':m['accuracy'],'precision':m['fake_precision'],'recall':m['fake_recall'],'f1':m['fake_f1'],
        'macro_f1':m['macro_f1'],'roc_auc':m['roc_auc'],'confusion_matrix':m['confusion_matrix']}}}
    result.update(config=config, dashboard=dashboard,
                  validation_improved=result['validate']['gated']['macro_f1'] > result['validate']['concatenation']['macro_f1'])
    return result


def default_model(gated, reports):
    """Promotion additionally requires a successful local verification tied to this export."""
    proof = Path(gated).parent / 'windows_integration_verification.json'
    if proof.is_file():
        saved = read_json(proof)
        if (saved.get('passed') and saved.get('run_id')==reports['config']['run_id']
            and saved.get('manifest_sha256')==file_sha256(Path(gated)/'sha256.json')
            and reports['validation_improved']):
            return 'gated_fusion'
    return 'clip_fusion'


class SixModelPredictor(FiveModelPredictor):
    def __init__(self, baseline_bundle, clip_bundle, gated_bundle):
        base, clip, gated = map(Path, (baseline_bundle,clip_bundle,gated_bundle))
        gated_signature(base,clip,gated)
        manifest = verify_bundle(gated)
        self.reports = verified_reports(base,clip,gated)
        config = self.reports['config']
        # All shared encoders and reference heads must be byte-identical to the export.
        for name, expected in manifest.items():
            if name.startswith(('encoders/', 'models/')):
                source = base/name if name.startswith('encoders/distilbert/') or name=='models/text.pt' else clip/name
                if not source.is_file() or file_sha256(source)!=expected:
                    raise ValueError(f'Cannot share mismatched encoder/reference weights: {source}')
        super().__init__(base,clip)
        state = torch.load(gated/'gated_fusion.pt',map_location='cpu',weights_only=True)
        if state['run_id']!=config['run_id'] or state['spec']!=config['spec'] or state['labels']!=config['labels'] or state['seed']!=config['seed']:
            raise ValueError('Gated checkpoint does not match configuration')
        if state['spec']['text_dim']!=ENCODER_CONFIG['text_dim'] or state['spec']['image_dim']!=self.clip.dimension:
            raise ValueError('Gated input dimensions differ from frozen encoders')
        self.gated_head = GatedFusion(**state['spec'])
        self.gated_head.load_state_dict(state['state_dict'],strict=True)
        self.gated_head.requires_grad_(False).eval()
        self.warning_policy = config['warning_policy']
        self.gated_run_id = config['run_id']

    @torch.inference_mode()
    def analyze(self,title,image):
        if not isinstance(title,str) or not title.strip():
            raise ValueError('Enter a headline')
        with self._lock:
            text=self.baseline.encoders.encode_text([title.strip()])
            resnet=self.baseline.encoders.encode_images([image])
            clip=self.clip.encode([image])
            logits={k:model(text,resnet) for k,model in self.baseline.heads.items()}
            logits.update({k:model(text,clip) for k,model in self.clip_heads.items()})
            logits['gated_fusion'],gt,gi=self.gated_head(text,clip)
            scores={}
            for kind,value in logits.items():
                if value.shape!=(1,2) or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid classifier output: {kind}')
                scores[kind]=value.softmax(1)[0].cpu().tolist()
            gates=[gt.item(),gi.item()]
            if not all(0<=v<=1 for v in gates) or not math.isclose(sum(gates),1,abs_tol=1e-6):
                raise ValueError('Invalid learned gate coefficients')
            disagreement=abs(scores['text'][0]-scores['clip_image'][0])
            high=disagreement>=self.warning_policy['threshold']
            return {'scores':scores,'g_text':gates[0],'g_image':gates[1], 'disagreement':disagreement,
                    'threshold':self.warning_policy['threshold'], 'high_disagreement':high,
                    'warning':self.warning_policy['message'] if high and self.warning_policy['enabled'] else None}

    def predict(self,title,image):
        return self.analyze(title,image)['scores']
