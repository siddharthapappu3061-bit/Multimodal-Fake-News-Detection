"""Validation-only threshold selection, fixed audit, then frozen test reporting."""
import sys
import json
import hashlib
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from reliability import selective_metrics,accepted_mask
from integration_support import read_json,file_sha256,bundle_paths
from gated_integration import gated_bundle_path,verify_bundle

OUT=Path(__file__).resolve().parent
RULES=dict(validation_partition='SHA256(reliability-v1:<id>), first 300 tune, remaining 200 audit',
    confidence_thresholds=[.55,.60,.65,.70,.75,.80,.85,.90,.95],
    disagreement_thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9],
    minimum_coverage=.70,minimum_accuracy_gain=.02,maximum_accepted_fake_recall_drop=.03,
    minimum_accepted_per_class=20,
    selection='Among eligible tuning candidates maximize accepted accuracy; then coverage; then prefer confidence. Audit chosen candidate once; no fallback search on audit/test.')


def read_split(gated,split):
    frame=pd.read_csv(gated/'results'/split/'predictions.csv',dtype={'id':str})
    original=pd.read_csv(ROOT/'data'/f'{split}_processed.csv',dtype={'id':str}).set_index('id')
    assert frame.id.is_unique and len(frame)==500 and set(frame.id)==set(original.index)
    assert np.array_equal(frame.label,original.loc[frame.id,'2_way_label'])
    frame['disagreement']=abs(frame.text_fake-frame.clip_image_fake)
    frame['confidence']=np.maximum(frame.gated_fake,1-frame.gated_fake)
    frame['gated_prediction']=np.where(frame.gated_fake>=.5,0,1)
    frame['correct']=frame.gated_prediction==frame.label
    frame['unimodal_agreement']=(frame.text_fake>=.5)==(frame.clip_image_fake>=.5)
    return frame


def evaluate(frame,strategy='none',threshold=None):
    return selective_metrics(frame.label,frame.gated_fake,frame.disagreement,strategy,threshold)


def useful(candidate,baseline):
    return (candidate['coverage']>=RULES['minimum_coverage']
        and candidate['accepted_accuracy'] is not None
        and candidate['accepted_accuracy']>=baseline['accepted_accuracy']+RULES['minimum_accuracy_gain']
        and candidate['accepted_fake_count']>=RULES['minimum_accepted_per_class']
        and candidate['accepted_real_count']>=RULES['minimum_accepted_per_class']
        and candidate['accepted_fake_recall']>=baseline['accepted_fake_recall']-RULES['maximum_accepted_fake_recall_drop'])


def select_policy(validation):
    # Neither test data nor individual diagnostic examples are accepted by this function.
    order=sorted(validation.id,key=lambda v:hashlib.sha256(('reliability-v1:'+v).encode()).hexdigest())
    indexed=validation.set_index('id',drop=False)
    tune,audit=indexed.loc[order[:300]],indexed.loc[order[300:]]
    rows=[]; baseline=evaluate(tune)
    for strategy in ['confidence','disagreement']:
        for threshold in RULES[strategy+'_thresholds']:
            score=evaluate(tune,strategy,threshold)
            rows.append(dict(strategy=strategy,threshold=threshold,eligible=useful(score,baseline),**score))
    eligible=[r for r in rows if r['eligible']]
    chosen=max(eligible,key=lambda r:(r['accepted_accuracy'],r['coverage'],r['strategy']=='confidence')) if eligible else None
    audit_score=evaluate(audit,chosen['strategy'],chosen['threshold']) if chosen else evaluate(audit)
    audit_pass=bool(chosen and useful(audit_score,evaluate(audit)))
    available=bool(chosen and audit_pass)
    return dict(available=available,strategy=chosen['strategy'] if chosen else 'none',threshold=chosen['threshold'] if chosen else None,
                audit_passed=audit_pass,selection_split='validate',rules=RULES,
                tuning_baseline=baseline,tuning_selected=chosen,audit_baseline=evaluate(audit),audit_selected=audit_score,
                tune_ids=order[:300],audit_ids=order[300:],
                reason='Passed predeclared tuning and validation-audit criteria.' if available else 'No policy passed both tuning and validation-audit criteria; informational warnings only.'),rows


def save(value,path):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    gated=gated_bundle_path(ROOT)
    verify_bundle(gated)
    if read_json(gated/'config.json')['labels'] != {'0':'Fake','1':'Real'}:
        raise ValueError('Unexpected label mapping')
    validation=read_split(gated,'validate')
    policy,rows=select_policy(validation)
    policy.update(run_id=read_json(gated/'config.json')['run_id'],manifest_sha256=file_sha256(gated/'sha256.json'))
    policy_path=OUT/'policy.json'
    # Reruns must not silently retune a frozen policy after test evaluation.
    if policy_path.exists() and read_json(policy_path)!=policy:
        raise ValueError('Frozen policy changed. Stop; do not retune after test inspection.')
    save(policy,policy_path)  # Freeze before opening any test predictions here.
    pd.DataFrame(rows).to_csv(OUT/'validation_tuning_candidates.csv',index=False)
    partition={v:'tuning' for v in policy['tune_ids']}; partition.update({v:'audit' for v in policy['audit_ids']})
    validation['policy_partition']=validation.id.map(partition)
    validation.to_csv(OUT/'validation_examples.csv',index=False)
    # Freeze one representative from each family using tuning only, for descriptive comparisons.
    family=[]
    for strategy in ['confidence','disagreement']:
        allowed=[r for r in rows if r['strategy']==strategy and r['coverage']>=RULES['minimum_coverage'] and r['accepted_fake_count']>=20 and r['accepted_real_count']>=20]
        if allowed:
            best=max(allowed,key=lambda r:(r['accepted_accuracy'],r['coverage']))
            family.append((strategy,best['threshold']))
    save({'selected':{'strategy':policy['strategy'],'threshold':policy['threshold'],'available':policy['available']},'comparison_policies':family},OUT/'frozen_policies.json')
    test=read_split(gated,'test')
    comparisons=[]
    for split,frame in [('validation_tuning',validation[validation.policy_partition=='tuning']),('validation_audit',validation[validation.policy_partition=='audit']),('validation_all',validation),('test',test)]:
        for strategy,threshold in [('none',None),*family]:
            comparisons.append(dict(split=split,strategy=strategy,threshold=threshold,**evaluate(frame,strategy,threshold)))
    pd.DataFrame(comparisons).to_csv(OUT/'comparison.csv',index=False)
    selected_strategy=policy['strategy'] if policy['available'] else 'none'
    selected_threshold=policy['threshold'] if policy['available'] else None
    test['accepted']=accepted_mask(test.gated_fake,test.disagreement,selected_strategy,selected_threshold)
    test['displayed_decision']=np.where(test.accepted,np.where(test.gated_prediction==0,'Fake','Real'),'Requires Verification')
    test.to_csv(OUT/'test_selected_predictions.csv',index=False)
    errors=~validation.correct
    warning_threshold=read_json(gated/'config.json')['warning_policy']['threshold']
    groups=[]
    for name,mask in [('Unimodal agree',validation.unimodal_agreement),('Unimodal disagree',~validation.unimodal_agreement),('Low disagreement',validation.disagreement<warning_threshold),('High disagreement',validation.disagreement>=warning_threshold)]:
        groups.append(dict(group=name,count=int(mask.sum()),errors=int(errors[mask].sum()),error_rate=float(errors[mask].mean())))
    association=dict(error_detection_roc_auc=float(roc_auc_score(errors,validation.disagreement)),
                     uncertainty_error_detection_roc_auc=float(roc_auc_score(errors,1-validation.confidence)),
                     disagreement_error_correlation=float(np.corrcoef(validation.disagreement,errors.astype(int))[0,1]),groups=groups,
                     warning_threshold=warning_threshold)
    report=dict(policy=policy,association=association,
                validation_before=evaluate(validation),validation_after=evaluate(validation,selected_strategy,selected_threshold),
                test_before=evaluate(test),test_after=evaluate(test,selected_strategy,selected_threshold),
                limits='No retraining or calibration. Thresholds use validation only; audit is disjoint from threshold tuning but original model selection already used validation. This is not an independent prospective guarantee. Test is descriptive only and never changes policy. Accepted Fake recall excludes abstained Fake cases; overall Fake recall counts abstentions as missed automatic detections.')
    save(report,OUT/'report.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar([g['group']+'\nn='+str(g['count']) for g in groups],[g['error_rate'] for g in groups]); ax.set_ylabel('Gated error rate (validation)')
    fig.tight_layout();fig.savefig(OUT/'disagreement_errors.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    bins=pd.cut(validation.disagreement,[0,.2,.4,.6,.8,1],include_lowest=True)
    summary=validation.assign(error=errors).groupby(bins,observed=True).agg(error_rate=('error','mean'),count=('error','size'))
    ax.bar([str(k)+'\nn='+str(n) for k,n in zip(summary.index,summary['count'])],summary.error_rate)
    ax.set_ylabel('Validation error rate');ax.set_xlabel('Disagreement bin');fig.tight_layout();fig.savefig(OUT/'disagreement_bins.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for strategy in ['confidence','disagreement']:
        points=sorted([r for r in rows if r['strategy']==strategy and r['accepted_count']],key=lambda r:r['coverage'])
        ax.plot([r['coverage'] for r in points],[1-r['accepted_accuracy'] for r in points],'o-',label=strategy)
    ax.set(xlabel='Coverage on validation tuning partition',ylabel='Accepted error rate');ax.legend();fig.tight_layout();fig.savefig(OUT/'risk_coverage.png');plt.close(fig)
    def pct(x): return 'undefined' if x is None else f'{100*x:.2f}%'
    lines=['# Reliability and selective classification report','',policy['reason'],'',
        'The underlying six classifiers and their probabilities are unchanged. Optional abstention applies only to the gated final display.','',
        f"Selected strategy: **{policy['strategy']}**; threshold: **{policy['threshold']}**; enabled as an option: **{policy['available']}**.",'',
        '## Before/after (accepted predictions only)','',
        '| Split / policy | Accepted accuracy | Coverage | Abstention | Abstained error | Accepted Fake recall | Overall Fake recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split in ['validation','test']:
        for phase in ['before','after']:
            m=report[split+'_'+phase]
            lines.append('| '+split+' '+phase+' | '+' | '.join(pct(m[k]) for k in ['accepted_accuracy','coverage','abstention_rate','abstained_error_rate','accepted_fake_recall','overall_fake_recall'])+' |')
    lines+=['','## Frozen candidate comparisons (not deployed)','',
        '| Split | Candidate | Accepted accuracy | Coverage | Abstention | Abstained error | Accepted Fake recall | Overall Fake recall |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for item in comparisons:
        if item['split'] in ['validation_audit','test'] and item['strategy']!='none':
            lines.append('| '+item['split']+' | '+item['strategy']+' '+str(item['threshold'])+' | '+' | '.join(pct(item[k]) for k in ['accepted_accuracy','coverage','abstention_rate','abstained_error_rate','accepted_fake_recall','overall_fake_recall'])+' |')
    lines+=['',f"The selected confidence candidate retained {pct(policy['audit_selected']['coverage'])} on the reserved audit, below the declared 70% coverage requirement. The later test results do not override this rejection.",'',
        '## Selection and limitations','',
        'Threshold grids and usefulness criteria were declared in code before evaluation. Validation IDs were deterministically partitioned into 300 tuning / 200 audit examples without changing any dataset files or model splits. Tuning chooses highest accepted accuracy subject to coverage ≥70%, gain ≥2 percentage points, at least 20 accepted examples per class and accepted Fake recall no more than 3 points below baseline. The selected policy must pass the same audit criteria; no second search is performed on audit failures. Test results cannot promote, reject or retune it.','',
        f"Disagreement error-detection AUC: **{association['error_detection_roc_auc']:.4f}**. Uncertainty (1 − maximum model probability) error-detection AUC: **{association['uncertainty_error_detection_roc_auc']:.4f}**. Disagreement is assessed empirically, not assumed reliable.",'',report['limits'],'',
        'Model probability is not factual certainty. High accepted accuracy describes a smaller subset and must be read together with coverage. Neither learned gates nor disagreement scores are exact feature importance or proof of misinformation.','',
        'Files: `validation_examples.csv`, `validation_tuning_candidates.csv`, `comparison.csv`, `frozen_policies.json`, `policy.json`, `report.json`, and `test_selected_predictions.csv`. Charts: `disagreement_errors.png`, `disagreement_bins.png`, `risk_coverage.png`.','',
        'The **Use selective classification for gated fusion** control is disabled for this rejected policy. The mechanism is implemented and tested, but only a compatible policy that passes validation checks can enable it. It is off by default when available; changing it or the selected model reuses the same model probabilities. Other classifiers are never subjected to this gated-specific threshold.','',
        'Rollback: restore `six_model_app.py` from `backups/pre_reliability_2026-10-20/six_model_app.py`; model files require no rollback.']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['association','validation_before','validation_after','test_before','test_after']},indent=2))
    print(policy['reason'],policy['strategy'],policy['threshold'])


if __name__=='__main__':
    main()
