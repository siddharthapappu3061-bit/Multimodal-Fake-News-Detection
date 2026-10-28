"""Measured sensitivity and disagreement diagnostics; never relabel samples."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, precision_recall_curve)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def metrics(y, fake):
    y, fake = np.asarray(y), np.asarray(fake)
    if len(y) == 0 or fake.shape != y.shape or not np.isfinite(fake).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Expected aligned finite binary predictions')
    if np.any((fake < 0) | (fake > 1)):
        raise ValueError('Probabilities must be in [0, 1]')
    pred = np.where(fake >= .5, 0, 1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'macro_f1': float(f.mean()),
            'fake_precision': float(p[0]), 'fake_recall': float(r[0]), 'fake_f1': float(f[0]),
            'real_precision': float(p[1]), 'real_recall': float(r[1]), 'real_f1': float(f[1]),
            'roc_auc': float(roc_auc_score(y == 0, fake)) if len(np.unique(y)) == 2 else None,
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist()}


@torch.inference_mode()
def forward_batches(model, data, device='cpu', batch_size=128):
    model.to(device).eval()
    probabilities, gates = [], []
    for start in range(0, len(data['text']), batch_size):
        out = model(data['text'][start:start + batch_size].to(device), data['image'][start:start + batch_size].to(device))
        if isinstance(out, tuple):
            logits, text_gate, image_gate = out
            gates.append(torch.stack([text_gate, image_gate], dim=1).cpu())
        else:
            logits = out
        probabilities.append(logits.softmax(1)[:, 0].cpu())
    return torch.cat(probabilities).numpy(), torch.cat(gates).numpy() if gates else None


def derangement(count, rng):
    if count < 2:
        raise ValueError('Swapping requires at least two distinct samples')
    # Random cycle: every row receives a different row's feature, with no label use.
    order = rng.permutation(count)
    mapping = np.empty(count, dtype=int)
    mapping[order] = np.roll(order, 1)
    return mapping


def ablations(model, data, training, output, device='cpu', seed=42, repetitions=5, noise_scale=.1):
    """Only the unperturbed pair retains a meaningful ground-truth evaluation."""
    if repetitions < 1 or noise_scale < 0:
        raise ValueError('Invalid diagnostic settings')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    normal, _ = forward_batches(model, data, device)
    y = data['labels'].numpy()
    ids, rows = data['ids'], []
    reference = {k: training[k].mean(0, keepdim=True).expand_as(data[k]) for k in ('text', 'image')}
    rng = np.random.default_rng(seed)

    def record(name, text, image, repetition=0, donor=None):
        changed, _ = forward_batches(model, {'text': text, 'image': image}, device)
        for i, sample_id in enumerate(ids):
            rows.append({'id': sample_id, 'experiment': name, 'repetition': repetition,
                         'donor_id': ids[donor[i]] if donor is not None else None,
                         'normal_fake': float(normal[i]), 'perturbed_fake': float(changed[i]),
                         'delta_fake': float(changed[i] - normal[i]), 'abs_delta_fake': float(abs(changed[i] - normal[i])),
                         'class_changed': bool((changed[i] >= .5) != (normal[i] >= .5))})

    for repeat in range(repetitions):
        # Identical donor mapping for text/image swaps enables paired sensitivity comparison.
        donor = derangement(len(ids), rng)
        record('swap_image', data['text'], data['image'][donor], repeat, donor)
        record('swap_text', data['text'][donor], data['image'], repeat, donor)
        for key in ('text', 'image'):
            noise = torch.tensor(rng.standard_normal(tuple(data[key].shape)), dtype=torch.float32)
            perturbed = data[key] + noise_scale * training[key].std(0, unbiased=False) * noise
            record('noise_' + key, perturbed if key == 'text' else data['text'],
                   perturbed if key == 'image' else data['image'], repeat)
    record('mean_image', data['text'], reference['image'])
    record('mean_text', reference['text'], data['image'])
    record('mean_both', reference['text'], reference['image'])
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'per_sample_interventions.csv', index=False)
    summary = frame.groupby('experiment').agg(mean_abs_probability_change=('abs_delta_fake', 'mean'),
        mean_signed_probability_change=('delta_fake', 'mean'), class_change_rate=('class_changed', 'mean')).reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    paired = frame[frame.experiment.isin(['swap_text', 'swap_image'])].pivot_table(
        index='id', columns='experiment', values='abs_delta_fake', aggfunc='mean')
    difference = (paired['swap_text'] - paired['swap_image']).to_numpy()
    boot = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(1000)])
    low, high = np.quantile(boot, [.025, .975])
    report = {'normal_metrics': metrics(y, normal), 'examples': len(ids), 'seed': seed,
              'repetitions': repetitions, 'noise_scale': noise_scale, 'interventions': summary.to_dict('records'),
              'swap_text_minus_image_abs_change': float(difference.mean()),
              'descriptive_bootstrap_95_interval': [float(low), float(high)],
              'interpretation': ('Text swaps were more disruptive in this diagnostic.' if low > 0 else
                                 'Image swaps were more disruptive in this diagnostic.' if high < 0 else
                                 'This diagnostic does not clearly separate text and image swap sensitivity.'),
              'limits': 'Swaps, Gaussian noise and training means may be out of distribution. No intervention accuracy is claimed because mismatched pairs lack justified labels. Sensitivity and gates are not causal feature importance.'}
    save_json(report, output / 'report.json')
    plt = plotting()
    summary.set_index('experiment')[['mean_abs_probability_change', 'class_change_rate']].plot.bar(figsize=(11, 4))
    plt.tight_layout(); plt.savefig(output / 'ablation.png'); plt.close()
    return report


def disagreement_score(text_fake, image_fake):
    text, image = np.asarray(text_fake, dtype=float), np.asarray(image_fake, dtype=float)
    if text.shape != image.shape or not np.isfinite(text).all() or not np.isfinite(image).all() or np.any((text < 0) | (text > 1) | (image < 0) | (image > 1)):
        raise ValueError('Expected aligned probabilities in [0, 1]')
    return np.abs(text - image)


def choose_threshold(text, image, split='validate', quantile=.8, fixed=None):
    if split != 'validate' or not 0 < quantile < 1:
        raise ValueError('Thresholds must be chosen from validation only')
    score = disagreement_score(text, image)
    threshold = float(np.quantile(score, quantile)) if fixed is None else float(fixed)
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be in [0, 1]')
    return {'threshold': threshold, 'source_split': 'validate', 'method': 'validation quantile' if fixed is None else 'predeclared user threshold',
            'quantile': quantile if fixed is None else None, 'validated_threshold': False}


def disagreement_report(y, text, image, fused, threshold, split='validate'):
    score = disagreement_score(text, image)
    y, fused = np.asarray(y), np.asarray(fused)
    metrics(y, fused)  # Validate aligned labels and fused probabilities too.
    if y.shape != score.shape:
        raise ValueError('Disagreement rows are not aligned')
    errors = (np.where(fused >= .5, 0, 1) != y)
    opposite = (np.asarray(text) >= .5) != (np.asarray(image) >= .5)
    high = score >= threshold['threshold']
    def group(mask):
        count = int(mask.sum())
        return {'count': count, 'error_rate': float(errors[mask].mean()) if count else None}
    report = {'split': split, 'examples': len(y), 'threshold_policy': threshold, 'class_disagreement_frequency': float(opposite.mean()),
              'high_score_frequency': float(high.mean()), 'class_agree': group(~opposite), 'class_disagree': group(opposite),
              'low_score': group(~high), 'high_score': group(high), 'overall_error_rate': float(errors.mean()),
              'error_detection_roc_auc': float(roc_auc_score(errors, score)) if len(np.unique(errors)) == 2 else None,
              'score_error_correlation': float(np.corrcoef(score, errors.astype(float))[0, 1]) if score.std() > 0 and errors.std() > 0 else None,
              'bins': [{'lower': float(a), 'upper': float(b), **group((score >= a) & ((score < b) if b < 1 else (score <= b)))}
                       for a, b in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:])]}
    return report


def warning_policy(validation_report):
    if validation_report['split'] != 'validate' or validation_report['threshold_policy']['source_split'] != 'validate':
        raise ValueError('Warning policy must use validation data')
    high, low = validation_report['high_score'], validation_report['low_score']
    auc = validation_report['error_detection_roc_auc']
    enabled = (high['count'] >= 20 and low['count'] >= 20 and high['error_rate'] > low['error_rate']
               and auc is not None and auc > .5)
    return {'enabled': bool(enabled), 'threshold': validation_report['threshold_policy']['threshold'],
            'criterion': 'Exploratory validation association: at least 20 cases per group, higher high-score error rate, error-detection AUC > 0.5; not statistical validation.',
            'message': 'Text and image models disagree substantially. The combined prediction may be less reliable.',
            'changes_label': False}


def gate_statistics(gates, y=None):
    gates = np.asarray(gates)
    if gates.ndim != 2 or gates.shape[1] != 2 or not np.isfinite(gates).all() or np.any((gates < 0) | (gates > 1)) or not np.allclose(gates.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid modality gates')
    return {name: {'mean': float(gates[:, i].mean()), 'std': float(gates[:, i].std()),
                   'quantiles': np.quantile(gates[:, i], [0, .25, .5, .75, 1]).tolist()}
            for i, name in enumerate(['text', 'image'])}


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def diagnostic_plots(y, probabilities, gates, history, disagreement, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    plt = plotting()
    fig, axes = plt.subplots(1, len(probabilities), figsize=(5 * len(probabilities), 4))
    for ax, (name, fake) in zip(np.atleast_1d(axes), probabilities.items()):
        matrix = np.array(metrics(y, fake)['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues')
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha='center')
        ax.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], xlabel='Predicted', ylabel='Actual')
    fig.tight_layout(); fig.savefig(output / 'confusion_matrices.png'); plt.close(fig)
    for name, fake in probabilities.items():
        precision, recall, _ = precision_recall_curve(np.asarray(y) == 0, fake)
        plt.plot(recall, precision, label=name)
    plt.xlabel('Fake recall'); plt.ylabel('Fake precision'); plt.legend(); plt.tight_layout(); plt.savefig(output / 'precision_recall.png'); plt.close()
    plt.hist(gates[:, 0], bins=20, alpha=.6, label='Text'); plt.hist(gates[:, 1], bins=20, alpha=.6, label='Image')
    plt.legend(); plt.xlabel('Gate coefficient (not feature importance)'); plt.tight_layout(); plt.savefig(output / 'gate_distribution.png'); plt.close()
    for key in ['train_loss', 'validation_loss']:
        plt.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Cross-entropy'); plt.tight_layout(); plt.savefig(output / 'loss_curves.png'); plt.close()
    bins = disagreement['bins']
    plt.bar([f"{b['lower']:.1f}–{b['upper']:.1f}" for b in bins], [b['error_rate'] if b['error_rate'] is not None else 0 for b in bins])
    plt.xlabel('Disagreement score bin (empty bins show 0)'); plt.ylabel('Fusion error rate'); plt.tight_layout(); plt.savefig(output / 'disagreement_errors.png'); plt.close()
