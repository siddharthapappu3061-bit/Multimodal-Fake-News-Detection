"""Display-only selective decisions. No model loading, training or probability mutation."""
import json
import hashlib
import math
from pathlib import Path
import numpy as np


def assess(text_fake, image_fake, gated_fake):
    values = np.asarray([text_fake, image_fake, gated_fake], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Expected finite probabilities in [0,1]')
    label = lambda p: 'Fake' if p >= .5 else 'Real'
    return dict(predicted_class=label(gated_fake), model_probability=max(gated_fake,1-gated_fake),
                text_prediction=label(text_fake), image_prediction=label(image_fake),
                disagreement=abs(text_fake-image_fake), unimodal_agreement=label(text_fake)==label(image_fake))


def accepted_mask(fake, disagreement, strategy='none', threshold=None):
    fake, disagreement = np.asarray(fake,dtype=float), np.asarray(disagreement,dtype=float)
    if fake.shape != disagreement.shape or not np.isfinite(fake).all() or not np.isfinite(disagreement).all() or np.any((fake<0)|(fake>1)|(disagreement<0)|(disagreement>1)):
        raise ValueError('Invalid aligned probabilities/disagreement')
    if strategy=='none':
        return np.ones(fake.shape,dtype=bool)
    if strategy=='combined':
        if not isinstance(threshold,dict) or set(threshold)!= {'confidence','disagreement'}:
            raise ValueError('Combined policy requires confidence and disagreement thresholds')
        return (accepted_mask(fake,disagreement,'confidence',threshold['confidence']) &
                accepted_mask(fake,disagreement,'disagreement',threshold['disagreement']))
    if threshold is None or not math.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Invalid abstention threshold')
    if strategy=='confidence':
        return np.maximum(fake,1-fake)>=threshold
    if strategy=='disagreement':
        return disagreement<threshold
    raise ValueError('Unknown selective strategy')


def selective_metrics(labels,fake,disagreement,strategy='none',threshold=None):
    labels,fake=np.asarray(labels),np.asarray(fake)
    if labels.shape!=fake.shape or not len(labels) or not np.isin(labels,[0,1]).all():
        raise ValueError('Expected aligned binary labels')
    accepted=accepted_mask(fake,disagreement,strategy,threshold)
    predicted=np.where(fake>=.5,0,1)
    correct=predicted==labels
    n=int(accepted.sum()); rejected=int((~accepted).sum())
    fake_accepted=(labels==0)&accepted
    real_accepted=(labels==1)&accepted
    fake_total=int((labels==0).sum())
    true_fake=int((fake_accepted & (predicted==0)).sum())
    return dict(count=len(labels),accepted_count=n,abstained_count=rejected,coverage=float(accepted.mean()),
        abstention_rate=float((~accepted).mean()),accepted_accuracy=float(correct[accepted].mean()) if n else None,
        accepted_error_rate=float((~correct[accepted]).mean()) if n else None,
        abstained_error_rate=float((~correct[~accepted]).mean()) if rejected else None,
        accepted_fake_count=int(fake_accepted.sum()),accepted_real_count=int(real_accepted.sum()),
        accepted_fake_recall=true_fake/int(fake_accepted.sum()) if fake_accepted.any() else None,
        overall_fake_recall=true_fake/fake_total if fake_total else None,
        fake_coverage=int(fake_accepted.sum())/fake_total if fake_total else None,
        real_coverage=int(real_accepted.sum())/int((labels==1).sum()) if (labels==1).any() else None)


def displayed_decision(scores,policy,enabled=False,primary='gated_fusion'):
    """Apply only to the gated display; original class/probabilities remain accessible."""
    values=scores[primary]
    label='Fake' if values[0]>=values[1] else 'Real'
    if not enabled or primary!='gated_fusion' or not policy.get('available',False):
        return label
    info=assess(scores['text'][0],scores['clip_image'][0],scores['gated_fusion'][0])
    accepted=accepted_mask(np.array([scores['gated_fusion'][0]]),np.array([info['disagreement']]),policy['strategy'],policy['threshold'])[0]
    return label if accepted else 'Requires Verification'


def load_policy(path,gated):
    """Fail closed to informational-only mode for absent, stale or malformed policy files."""
    fallback={'available':False,'reason':'No compatible validation-checked abstention policy is available.'}
    path,gated=Path(path),Path(gated)
    try:
        policy=json.loads(path.read_text(encoding='utf-8'))
        manifest=hashlib.sha256((gated/'sha256.json').read_bytes()).hexdigest()
        config=json.loads((gated/'config.json').read_text())
        if policy['run_id']!=config['run_id'] or policy['manifest_sha256']!=manifest or policy['selection_split']!='validate':
            return fallback
        if policy['available']:
            if policy['strategy'] not in ['confidence','disagreement','combined'] or not policy['audit_passed']:
                return fallback
            accepted_mask(np.array([.5]),np.array([.5]),policy['strategy'],policy['threshold'])
        return policy
    except (OSError,ValueError,KeyError,TypeError):
        return fallback


def policy_description(policy):
    """Human-readable acceptance rule, shared by the UI and reports."""
    strategy,threshold=policy['strategy'],policy['threshold']
    if strategy=='combined':
        return f"model probability ≥ {threshold['confidence']:.2f} and disagreement < {threshold['disagreement']:.2f}"
    if strategy=='confidence':
        return f"model probability ≥ {threshold:.2f}"
    if strategy=='disagreement':
        return f"disagreement < {threshold:.2f}"
    return 'all predictions accepted'
