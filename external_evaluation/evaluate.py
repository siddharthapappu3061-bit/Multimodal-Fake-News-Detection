"""Offline, independent evaluation. Never fits models or selects thresholds."""
import argparse
import csv
import hashlib
import json
import io
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score, confusion_matrix
from reliability import selective_metrics, accepted_mask, load_policy

FIELDS = ['sample_id', 'headline', 'image_path', 'verified_label', 'original_source_url',
          'verification_evidence_url', 'verification_notes']
OPTIONAL = ['association_status', 'association_evidence_url', 'association_notes']
KEYS = ['text', 'image', 'fusion', 'clip_image', 'clip_fusion', 'gated_fusion']


def json_default(value):
    """Preserve NumPy numeric types as JSON numbers, never stringified metrics."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def json_text(value):
    return json.dumps(value, indent=2, allow_nan=False, default=json_default)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def url_ok(value):
    try:
        parsed = urlparse(value)
        return parsed.scheme in ('http', 'https') and bool(parsed.hostname) and not any(c.isspace() for c in value)
    except ValueError:
        return False


def validate_dataset(path, project=ROOT):
    """Validate declared provenance; URLs/notes are not automated fact verification."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    encoding_warnings = []
    try:
        decoded = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        try:
            decoded = raw.decode('cp1252')
        except UnicodeDecodeError as exc:
            raise ValueError('CSV is neither UTF-8 nor Windows-1252. Save it as CSV UTF-8 and retry.') from exc
        encoding_warnings.append('CSV decoded as Windows-1252 because UTF-8 decoding failed. '
                                 'Check punctuation in the source; CSV UTF-8 is recommended. The file was not modified.')
    with io.StringIO(decoded, newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(FIELDS) <= set(reader.fieldnames):
            raise ValueError('Required CSV columns: ' + ', '.join(FIELDS))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Duplicate CSV column names')
        rows = list(reader)
    if not rows:
        raise ValueError('No external samples: the template intentionally contains headers only.')
    known_ids, known_titles = set(), set()
    for split in ('train', 'validate', 'test'):
        source = Path(project) / 'data' / f'{split}_processed.csv'
        if source.exists():
            with source.open(encoding='utf-8-sig', newline='') as stream:
                for item in csv.DictReader(stream):
                    known_ids.add(item['id'])
                    known_titles.add(' '.join(item.get('clean_title', '').lower().split()))
    errors, warnings, ids, pairs = [], encoding_warnings, set(), set()
    images, titles = set(), set()
    for number, row in enumerate(rows, 2):
        if None in row:
            raise ValueError(f'Row {number}: extra CSV fields; quote headlines/notes containing commas')
        row = {k: (v or '').strip() for k, v in row.items() if k is not None}
        prefix = f'Row {number}: '
        if any(not row.get(k) for k in FIELDS):
            errors.append(prefix + 'all required fields must be nonempty')
        if row.get('sample_id') in ids:
            errors.append(prefix + 'duplicate sample ID')
        ids.add(row.get('sample_id'))
        if row.get('verified_label') not in ('Fake', 'Real'):
            errors.append(prefix + 'verified_label must be exactly Fake or Real')
        for key in ('original_source_url', 'verification_evidence_url'):
            if not url_ok(row.get(key, '')):
                errors.append(prefix + f'invalid or missing {key}')
        title = ' '.join(row.get('headline', '').lower().split())
        if row.get('sample_id') in known_ids or title in known_titles:
            errors.append(prefix + 'overlaps local Fakeddit ID/headline; use independent samples')
        status = row.get('association_status', '')
        if status not in ('', 'not_established', 'supported_misleading'):
            errors.append(prefix + 'invalid association_status')
        if status == 'supported_misleading' and (not url_ok(row.get('association_evidence_url', '')) or not row.get('association_notes')):
            errors.append(prefix + 'misleading association needs explicit evidence URL and notes')
        raw = row.get('image_path', '').replace('\\', '/')
        target = Path(raw)
        if not target.is_absolute():
            target = path.parent / target
        target = target.resolve()
        try:
            with Image.open(target) as img:
                rgb = img.convert('RGB')
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            pair = (title, pixel_hash)
            if pair in pairs:
                errors.append(prefix + 'duplicate headline-image pair (including conflicting labels)')
            pairs.add(pair)
            if pixel_hash in images or title in titles:
                warnings.append(prefix + 'repeated image or headline; review event/claim dependence')
            images.add(pixel_hash)
            titles.add(title)
            row['image_sha256'] = digest(target)
            row['pixel_sha256'] = pixel_hash
        except (OSError, ValueError) as exc:
            errors.append(prefix + f'missing/unreadable image {target}: {exc}')
        row['resolved_image_path'] = str(target)
        rows[number - 2] = row
    if errors:
        raise ValueError('\n'.join(errors))
    return rows, warnings


def metrics(labels, fake):
    fake = np.asarray(fake, dtype=float)
    if fake.shape != (len(labels),) or not np.isfinite(fake).all() or ((fake < 0) | (fake > 1)).any():
        raise ValueError('Invalid model probabilities')
    predicted = np.where(fake >= .5, 0, 1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, pos_label=0, average='binary', zero_division=0)
    return dict(accuracy=float(accuracy_score(labels, predicted)), precision=float(precision), recall=float(recall),
                fake_f1=float(f1), macro_f1=float(f1_score(labels, predicted, labels=[0, 1], average='macro', zero_division=0)),
                roc_auc=float(roc_auc_score(np.asarray(labels) == 0, fake)) if len(set(labels)) == 2 else None,
                confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist())


def summarize(rows, outputs, policy, benchmark):
    if len(rows) != len(outputs) or not rows:
        raise ValueError('Expected one result per external sample')
    labels = np.array([0 if r['verified_label'] == 'Fake' else 1 for r in rows])
    probs = {}
    for key in KEYS:
        values = np.asarray([o['scores'][key] for o in outputs])
        if values.shape != (len(rows), 2) or not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f'Invalid two-class output: {key}')
        probs[key] = values[:, 0]
    report = {'count': len(rows), 'label_mapping': {'Fake': 0, 'Real': 1}, 'models': {},
              'class_counts': {'Fake': int((labels == 0).sum()), 'Real': int((labels == 1).sum())},
              'fakeddit_test_count': benchmark.get('count', 500),
              'roc_auc_note': 'Undefined (null) for a single-label dataset.', 'policy': policy,
              'benchmark_note': 'Different datasets and sampling; differences are descriptive, not a controlled significance test.'}
    predicted = {k: np.where(p >= .5, 0, 1) for k, p in probs.items()}
    d = abs(probs['text'] - probs['clip_image'])
    report['disagreement'] = {'mean_absolute_text_clip': float(d.mean()),
        'text_clip_class_disagreement_rate': float(np.mean(predicted['text'] != predicted['clip_image'])),
        'pairwise_class_disagreement_rates': {a + '__' + b: float(np.mean(predicted[a] != predicted[b]))
            for i, a in enumerate(KEYS) for b in KEYS[i+1:]}}
    for key in KEYS:
        m = metrics(labels, probs[key])
        base = benchmark['models'][key]
        report['models'][key] = {'external': m, 'fakeddit_test': base,
            'accuracy_difference': m['accuracy'] - base['accuracy'],
            'without_abstention': selective_metrics(labels, probs[key], d)}
    strategy = policy['strategy'] if policy.get('available') else 'none'
    threshold = policy.get('threshold') if policy.get('available') else None
    report['gated_selective'] = selective_metrics(labels, probs['gated_fusion'], d, strategy, threshold)
    report['gated_selective']['applied_strategy'] = strategy
    accepted = accepted_mask(probs['gated_fusion'], d, strategy, threshold)
    records, errors, fusion_counts = [], [], {}
    for i, row in enumerate(rows):
        for key in KEYS:
            pred, true = int(predicted[key][i]), int(labels[i])
            record = {**row, 'model': key, 'fake_probability': float(probs[key][i]),
                'real_probability': float(outputs[i]['scores'][key][1]), 'predicted_label': 'Fake' if pred == 0 else 'Real',
                'displayed_decision': 'Requires Verification' if key == 'gated_fusion' and not accepted[i] else ('Fake' if pred == 0 else 'Real'),
                'correct': pred == true, 'false_positive': pred == 0 and true == 1,
                'false_negative': pred == 1 and true == 0,
                'overconfident_error': pred != true and max(probs[key][i], 1-probs[key][i]) >= .95,
                'text_clip_disagree': bool(predicted['text'][i] != predicted['clip_image'][i]),
                'absolute_disagreement': float(d[i]),
                'evidence_supported_misleading_association': row.get('association_status') == 'supported_misleading'}
            records.append(record)
            if pred != true or record['text_clip_disagree'] or record['evidence_supported_misleading_association']:
                errors.append(record)
    fusion_cases = []
    for fusion, image in [('fusion', 'image'), ('clip_fusion', 'clip_image'), ('gated_fusion', 'clip_image')]:
        good = predicted[fusion] == labels
        text_good, image_good = predicted['text'] == labels, predicted[image] == labels
        flags = {'improves_both': good & ~text_good & ~image_good, 'worse_than_both': ~good & text_good & image_good,
                 'rescues_text': good & ~text_good, 'rescues_image': good & ~image_good,
                 'harms_text': ~good & text_good, 'harms_image': ~good & image_good}
        fusion_counts[fusion] = {k: int(v.sum()) for k, v in flags.items()}
        for i, row in enumerate(rows):
            active = [k for k, mask in flags.items() if mask[i]]
            if active:
                fusion_cases.append({'sample_id': row['sample_id'], 'model': fusion, 'categories': ';'.join(active)})
    report['fusion_comparisons'] = fusion_counts
    report['error_counts'] = {key: {flag: int(sum(r[flag] for r in records if r['model'] == key))
        for flag in ('false_positive', 'false_negative', 'overconfident_error')} for key in KEYS}
    return report, records, errors, fusion_cases


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    csv_sha256 = digest(args.csv)
    rows, warnings = validate_dataset(args.csv)
    if args.validate_only:
        print(json.dumps({'valid_samples': len(rows), 'warnings': warnings}, indent=2))
        return
    from integration_support import bundle_paths
    from gated_integration import SixModelPredictor, gated_bundle_path
    base, clip = bundle_paths(ROOT)
    gated = gated_bundle_path(ROOT)
    policy_path = ROOT / 'experiments/reliability_combined/policy.json'
    policy = load_policy(policy_path, gated)
    predictor = SixModelPredictor(base, clip, gated)
    outputs = []
    for i, row in enumerate(rows, 1):
        if digest(row['resolved_image_path']) != row['image_sha256']:
            raise ValueError('Image changed after validation')
        with Image.open(row['resolved_image_path']) as image:
            outputs.append(predictor.analyze(row['headline'], image.convert('RGB')))
        print(f'Inferred {i}/{len(rows)}', flush=True)
    report, records, errors, fusion = summarize(rows, outputs, policy, predictor.reports['dashboard'])
    if digest(args.csv) != csv_sha256:
        raise ValueError('CSV changed during evaluation; rerun with a stable dataset')
    out = ROOT / 'external_evaluation/results' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    report['warnings'] = warnings
    report['provenance'] = {'csv': str(args.csv.resolve()), 'csv_sha256': csv_sha256,
        'policy_sha256': digest(policy_path), 'baseline_clip_runs': predictor.run_ids, 'gated_run': predictor.gated_run_id,
        'script_sha256': digest(__file__), 'gated_manifest_sha256': digest(gated / 'sha256.json')}
    # Preserve completed inference before report export, including if a later export fails.
    (out / 'raw_outputs.json').write_text(json_text(outputs), encoding='utf-8')
    (out / 'metrics.json').write_text(json_text(report), encoding='utf-8')
    fields = list(dict.fromkeys(k for row in records for k in row))
    write_csv(out / 'predictions.csv', records, fields)
    write_csv(out / 'error_analysis.csv', errors, fields)
    write_csv(out / 'fusion_cases.csv', fusion, ['sample_id', 'model', 'categories'])
    lines = ['# External evaluation', '', f'Samples: {len(rows)}. Labels are supplied by human verification, not model predictions.',
        'No thresholds were fitted on these examples. Model probability is not factual certainty.', '',
        '| Model | Accuracy | Macro-F1 | ROC-AUC | Fakeddit accuracy |', '|---|---:|---:|---:|---:|']
    for key, values in report['models'].items():
        m = values['external']
        lines.append(f"| {key} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['roc_auc']} | {values['fakeddit_test']['accuracy']:.4f} |")
    lines += ['', '## Selective classification', json_text(report['gated_selective']),
        '', '## Error analysis', 'See error_analysis.csv for false positives (Real predicted Fake), false negatives, errors at probability >=0.95, and disagreements.',
        'See fusion_cases.csv for improvements/regressions against each unimodal model. Categories overlap and are not causal attributions.',
        'Misleading associations are included only when explicitly annotated with evidence and notes. Disagreement or an unrelated image alone does not establish a false claim.',
        'Confusion matrices in metrics.json have rows=true, columns=predicted, ordered [Fake, Real]. Undefined ROC-AUC is null; zero-denominator precision/recall/F1 uses zero.',
        'This external sample and Fakeddit differ in selection and domain. Do not interpret metric differences as a controlled experiment.',
        'No validated abstention is deployed when policy.available is false; coverage then stays 100%.']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
