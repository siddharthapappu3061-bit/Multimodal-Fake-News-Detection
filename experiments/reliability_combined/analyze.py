"""Validation-only comparison of three abstention families; test outcomes never read."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mpl-cache'))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from reliability import selective_metrics,accepted_mask,policy_description
from integration_support import read_json,file_sha256

OUT=Path(__file__).resolve().parent
GATED=ROOT/'experiments/gated_fusion/imported/windows_export/gated_artifacts'
# Declared before reading results; unchanged criteria and grids from the prior experiment.
RULES=dict(confidence_thresholds=[.55,.60,.65,.70,.75,.80,.85,.90,.95],
           disagreement_thresholds=[.1,.2,.3,.4,.5,.6,.7,.8,.9],
           minimum_coverage=.70,minimum_accuracy_gain=.02,maximum_accepted_fake_recall_drop=.03,
           minimum_accepted_per_class=20,partition='SHA256(reliability-v1:<id>): first 300 tuning, last 200 audit',
           selection='Highest eligible tuning selective accuracy, then coverage, then prefer confidence, disagreement, combined. Freeze one winner globally before audit; no replacement after audit failure.')


def save(value,path):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def evaluate(frame,strategy='none',threshold=None):
    return selective_metrics(frame.label,frame.gated_fake,frame.disagreement,strategy,threshold)


def failures(m,baseline):
    out=[]
    if m['coverage']<RULES['minimum_coverage']: out.append('Coverage below 70%')
    if m['accepted_accuracy'] is None or m['accepted_accuracy']<baseline['accepted_accuracy']+.02: out.append('Accepted accuracy gain below 2 percentage points')
    if min(m['accepted_fake_count'],m['accepted_real_count'])<20: out.append('Fewer than 20 accepted examples in a class')
    if m['accepted_fake_recall'] is None or m['accepted_fake_recall']<baseline['accepted_fake_recall']-.03: out.append('Accepted Fake recall falls by more than 3 percentage points')
    return out


def partition(frame):
    ids=sorted(frame.id,key=lambda value:hashlib.sha256(('reliability-v1:'+value).encode()).hexdigest())
    indexed=frame.set_index('id',drop=False)
    return indexed.loc[ids[:300]],indexed.loc[ids[300:]]


def tune(frame):
    """Receives tuning rows only; cannot inspect audit or test results."""
    baseline=evaluate(frame)
    settings=[('confidence',v) for v in RULES['confidence_thresholds']]
    settings += [('disagreement',v) for v in RULES['disagreement_thresholds']]
    settings += [('combined',{'confidence':c,'disagreement':d}) for c in RULES['confidence_thresholds'] for d in RULES['disagreement_thresholds']]
    candidates=[]
    for strategy,threshold in settings:
        score=evaluate(frame,strategy,threshold)
        candidates.append(dict(strategy=strategy,threshold=threshold,eligible=not failures(score,baseline),**score))
    rank=lambda r:(r['accepted_accuracy'] if r['accepted_accuracy'] is not None else -1,r['coverage'],{'confidence':2,'disagreement':1,'combined':0}[r['strategy']])
    family={}
    for strategy in ['confidence','disagreement','combined']:
        eligible=[r for r in candidates if r['strategy']==strategy and r['eligible']]
        if eligible: family[strategy]=max(eligible,key=rank)
    chosen=max(family.values(),key=rank) if family else None
    return candidates,family,chosen


def read_validation():
    config=read_json(GATED/'config.json');manifest=read_json(GATED/'sha256.json')
    if config['labels']!={'0':'Fake','1':'Real'}: raise ValueError('Unexpected label mapping')
    for name in ['config.json','gated_fusion.pt','results/validate/predictions.csv']:
        if file_sha256(GATED/name)!=manifest[name]: raise ValueError('Artifact mismatch: '+name)
    frame=pd.read_csv(GATED/'results/validate/predictions.csv',dtype={'id':str})
    original=pd.read_csv(ROOT/'data/validate_processed.csv',dtype={'id':str}).set_index('id')
    if len(frame)!=500 or not frame.id.is_unique or set(frame.id)!=set(original.index): raise ValueError('Wrong validation IDs')
    if not np.array_equal(frame.label,original.loc[frame.id,'2_way_label']): raise ValueError('Wrong validation labels')
    frame['disagreement']=abs(frame.text_fake-frame.clip_image_fake)
    frame['confidence']=np.maximum(frame.gated_fake,1-frame.gated_fake)
    frame['correct']=np.where(frame.gated_fake>=.5,0,1)==frame.label
    frame['unimodal_agreement']=(frame.text_fake>=.5)==(frame.clip_image_fake>=.5)
    evaluate(frame) # Validate probability ranges before selection.
    if not np.isfinite(frame[['text_fake','clip_image_fake']]).all().all() or not frame[['text_fake','clip_image_fake']].apply(lambda c:c.between(0,1).all()).all():
        raise ValueError('Invalid unimodal probabilities')
    return frame,config


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame,config=read_validation()
    tuning,audit=partition(frame)
    candidates,family,chosen=tune(tuning)
    frozen=dict(rules=RULES,run_id=config['run_id'],manifest_sha256=file_sha256(GATED/'sha256.json'),
                validation_predictions_sha256=file_sha256(GATED/'results/validate/predictions.csv'),
                tuning_ids=tuning.id.tolist(),audit_ids=audit.id.tolist(),family=family,chosen=chosen)
    frozen_path=OUT/'frozen_selection.json'
    if frozen_path.exists() and read_json(frozen_path)!=frozen: raise ValueError('Frozen selection changed; stop instead of retuning.')
    save(frozen,frozen_path) # Freeze before evaluating any audit outcomes.
    audit_baseline=evaluate(audit)
    selected_audit=evaluate(audit,chosen['strategy'],chosen['threshold']) if chosen else audit_baseline
    rejected=failures(selected_audit,audit_baseline) if chosen else ['No tuning candidate met usefulness criteria']
    available=bool(chosen and not rejected)
    policy=dict(available=available,strategy=chosen['strategy'] if chosen else 'none',threshold=chosen['threshold'] if chosen else None,
                audit_passed=available,selection_split='validate',run_id=config['run_id'],manifest_sha256=frozen['manifest_sha256'],
                version='combined-v2',audit_failures=rejected,
                reason='Selected policy passed the stated validation checks.' if available else 'Selected policy did not pass validation checks; informational warnings only.')
    save(policy,OUT/'policy.json')
    comparisons=[]
    for split,data in [('validation_tuning',tuning),('validation_audit',audit),('validation_all',frame)]:
        comparisons.append(dict(split=split,strategy='none',threshold=None,**evaluate(data)))
        for strategy,item in family.items():
            comparisons.append(dict(split=split,strategy=strategy,threshold=item['threshold'],**evaluate(data,strategy,item['threshold'])))
    pd.DataFrame(comparisons).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'tuning_candidates.csv',index=False)
    deployed_strategy=policy['strategy'] if available else 'none';deployed_threshold=policy['threshold'] if available else None
    frame['partition']=np.where(frame.id.isin(tuning.id),'tuning','audit')
    frame['accepted']=accepted_mask(frame.gated_fake,frame.disagreement,deployed_strategy,deployed_threshold)
    frame['displayed_decision']=np.where(frame.accepted,np.where(frame.gated_fake>=.5,'Fake','Real'),'Requires Verification')
    frame.to_csv(OUT/'validation_examples.csv',index=False)
    errors=~frame.correct
    threshold=config['warning_policy']['threshold']
    groups=[]
    for name,mask in [('Unimodal agreement',frame.unimodal_agreement),('Unimodal disagreement',~frame.unimodal_agreement),('Low disagreement',frame.disagreement<threshold),('High disagreement',frame.disagreement>=threshold)]:
        groups.append(dict(group=name,count=int(mask.sum()),error_rate=float(errors[mask].mean())))
    association=dict(error_detection_auc=float(roc_auc_score(errors,frame.disagreement)),
                     uncertainty_error_detection_auc=float(roc_auc_score(errors,1-frame.confidence)),
                     correlation=float(np.corrcoef(frame.disagreement,errors.astype(int))[0,1]),groups=groups)
    report=dict(policy=policy,association=association,comparisons=comparisons,
                validation_before=evaluate(frame),validation_after=evaluate(frame,deployed_strategy,deployed_threshold),
                audit_baseline=audit_baseline,selected_audit=selected_audit,
                limits='Validation was already used for model selection and an earlier policy study. This reused 300/200 partition is not fresh independent evidence. Multiple candidates were compared on tuning rows; audit cannot replace the frozen winner. No test outcomes were read for this analysis. Probabilities are uncalibrated model estimates, not factual verification.')
    save(report,OUT/'report.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar([g['group']+'\nn='+str(g['count']) for g in groups],[g['error_rate'] for g in groups]);ax.set_ylabel('Validation gated error rate')
    fig.tight_layout();fig.savefig(OUT/'disagreement_errors.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for strategy in ['confidence','disagreement','combined']:
        pts=[r for r in candidates if r['strategy']==strategy and r['accepted_accuracy'] is not None]
        ax.scatter([r['coverage'] for r in pts],[1-r['accepted_accuracy'] for r in pts],label=strategy,alpha=.6,s=18)
    ax.set(xlabel='Coverage on validation tuning rows',ylabel='Accepted error rate');ax.legend();fig.tight_layout();fig.savefig(OUT/'risk_coverage.png');plt.close(fig)
    def pct(v):return '—' if v is None else f'{100*v:.2f}%'
    lines=['# Validation reliability assessment: combined strategy','',policy['reason'],'',
        'No models were retrained. No external APIs or new datasets were used. All thresholds are selected from validation tuning rows only; the test set is not evaluated by this analysis.','',
        '## Validation comparison','',
        '| Partition | Strategy | Acceptance rule | Coverage | Selective accuracy | Abstention | Abstained error | Accepted Fake recall | Overall Fake recall |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append('| '+r['split']+' | '+r['strategy']+' | '+policy_description(r)+' | '+' | '.join(pct(r[k]) for k in ['coverage','accepted_accuracy','abstention_rate','abstained_error_rate','accepted_fake_recall','overall_fake_recall'])+' |')
    lines+=['','## Decision','',f"Frozen winner: **{policy['strategy']}**, acceptance rule **{policy_description(policy)}**.",
        'Audit failures: '+('; '.join(rejected) if rejected else 'none')+'.',
        'Only the frozen global winner can be enabled. Family candidates are shown for comparison, not selected after inspecting audit results.','',
        '## Disagreement findings','',f"Disagreement error-detection AUC: **{association['error_detection_auc']:.4f}**; uncertainty AUC: **{association['uncertainty_error_detection_auc']:.4f}**; disagreement/error correlation: **{association['correlation']:.4f}**.",
        'Higher disagreement is associated with more errors in these data, but its error-detection AUC is weak. It is not proof that an individual prediction is wrong.','',
        '## Method and limitations','',
        'Keep the existing deterministic partition: 300 tuning / 200 audit. Confidence grid: 0.55–0.95 in 0.05 increments; disagreement grid: 0.1–0.9 in 0.1 increments. Combined evaluates all 81 pairs and accepts only when confidence ≥ its threshold AND disagreement < its threshold. Low confidence OR high disagreement triggers abstention.',
        'The unchanged usefulness criteria require coverage ≥70%, selective accuracy gain ≥2 percentage points, at least 20 accepted examples per class and accepted Fake recall declining by no more than 3 percentage points. Choose highest eligible tuning accuracy, breaking ties by coverage then simpler strategy. Freeze before audit; never replace a failed winner using audit or test outcomes.','',report['limits'],'',
        'Selective accuracy applies only to accepted examples and is not directly interchangeable with full-coverage accuracy. Accepted Fake recall excludes abstained Fake cases; overall Fake recall counts abstentions as missed automatic detections. Undefined empty-group rates are shown as —.','',
        '## UI and running','',
        'Gated fusion remains default. All six comparisons remain accessible. The optional selective-classification checkbox is available only when the saved policy passed the validation checks; otherwise it is disabled and the existing binary decision and informational warnings remain. Requires Verification is a display decision only; the raw class and unchanged model probabilities remain visible. Gate coefficients are not exact feature importance.','',
        'From PowerShell:','', '```powershell',
        'cd "C:\\Users\\siddh\\OneDrive\\Desktop\\Multimodal_Fake_News_Detection"',
        '.\\CLIP-vision\\.venv-clip\\Scripts\\python.exe -m streamlit run app.py','```','',
        'Run tests with `.\\CLIP-vision\\.venv-clip\\Scripts\\python.exe tests/test_reliability_combined.py`. Reproduce this analysis with `.\\CLIP-vision\\.venv-clip\\Scripts\\python.exe experiments/reliability_combined/analyze.py`; changing a frozen selection is rejected.','',
        'Changed application files: `reliability.py`, `six_model_app.py`. New analysis/tests: `experiments/reliability_combined/analyze.py`, `tests/test_reliability_combined.py`. Earlier reliability results and training notebooks remain untouched. Backups: `backups/pre_reliability_combined_2026-10-23/`.','',
        'Results: `policy.json`, `frozen_selection.json`, `report.json`, `comparison.csv`, `validation_examples.csv`, `tuning_candidates.csv`, `disagreement_errors.png`, `risk_coverage.png`. Test and preservation evidence are saved alongside them.']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(policy=policy,association=association,validation_audit=[r for r in comparisons if r['split']=='validation_audit'],validation_all=[r for r in comparisons if r['split']=='validation_all']),indent=2))


if __name__=='__main__':main()
