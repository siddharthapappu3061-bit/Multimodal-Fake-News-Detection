"""Read-only cache loading, recoverable GPU head training and held-out evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from pipeline import atomic_save, gpu, fingerprint
from clip_experiment import read_cached_modality, digest, sha_file
from clip_models import ClipHead, FEATURE_SPEC
from models import Head
from gated_model import GatedFusion
from gated_analysis import (save_json, metrics, forward_batches, gate_statistics, disagreement_report,
                            warning_policy, diagnostic_plots, ablations)

FARGO_TITLE = 'A Fargo, North Dakota, man was arrested for clearing snow with a flamethrower.'


def load_caches(frames, baseline_bundle, clip_bundle, text_cache_root, clip_cache_root, fargo_image=None):
    base = json.loads((Path(baseline_bundle) / 'config.json').read_text())
    clip = json.loads((Path(clip_bundle) / 'config.json').read_text())
    if clip['baseline_run_id'] != base['run_id'] or base['labels'] != {'0': 'Fake', '1': 'Real'} or clip['labels'] != base['labels']:
        raise ValueError('Baseline/CLIP runs or label mappings disagree')
    if clip['feature_spec'] != FEATURE_SPEC or fingerprint(frames, base['encoders']) != base['feature_id']:
        raise ValueError('Data/encoder fingerprint differs from the original experiment')
    folders = {(split, kind): Path(root) / feature_id / split / ('text' if kind == 'text' else '')
               for split in frames for kind, root, feature_id in [('text', text_cache_root, base['feature_id']), ('image', clip_cache_root, clip['clip_feature_id'])]}
    missing = [str(path / '*.pt') for path in folders.values() if not list(path.glob('*.pt'))]
    if missing:
        raise FileNotFoundError('Required original cache shards are missing; no extraction was attempted:\n' + '\n'.join(missing))
    data, dims = {}, {}
    for split, frame in frames.items():
        data[split] = {'ids': frame.id.tolist(), 'labels': torch.tensor(frame['2_way_label'].tolist(), dtype=torch.long)}
        for kind in ('text', 'image'):
            folder = folders[split, kind]
            sample = torch.load(next(iter(sorted(folder.glob('*.pt')))), map_location='cpu', weights_only=True)
            dimension = sample['features'].shape[1]
            if kind in dims and dims[kind] != dimension:
                raise ValueError('Dimensions differ between splits')
            dims[kind] = dimension
            if kind == 'image':
                for path in folder.glob('*.pt'):
                    if torch.load(path, map_location='cpu', weights_only=True)['feature_id'] != clip['clip_feature_id']:
                        raise ValueError(f'Wrong CLIP shard fingerprint: {path}')
            tensor = read_cached_modality(folder, frame, dimension)
            data[split][kind] = tensor
            if kind == 'text':
                expected = clip['provenance']['baseline']['cache_sha256'][split]['text']
                if hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest() != expected:
                    raise ValueError(f'{split}: cached text differs from CLIP training provenance')
    if dims != {'text': clip['heads']['clip_fusion']['text_dim'], 'image': clip['heads']['clip_fusion']['image_dim']}:
        raise ValueError('Measured feature dimensions disagree with trained fusion checkpoint')
    normalize = lambda title: ''.join(c for c in title.casefold() if c.isalnum())
    if any(normalize(title) == normalize(FARGO_TITLE) for frame in frames.values() for title in frame.clean_title):
        raise ValueError('Fargo diagnostic appears in a dataset split. Stop; do not silently drop or train on it.')
    if fargo_image is not None:
        if not Path(fargo_image).is_file():
            raise FileNotFoundError(f'Missing Fargo image: {fargo_image}')
        fargo_sha = sha_file(fargo_image)
        if any(fargo_sha in set(frame.image_sha256) for frame in frames.values()):
            raise ValueError('Fargo diagnostic image is in the dataset; stop without modifying splits')
    return data, base, clip, dims


def existing_heads(baseline_bundle, clip_bundle):
    base = json.loads((Path(baseline_bundle) / 'config.json').read_text())
    clip = json.loads((Path(clip_bundle) / 'config.json').read_text())
    result = {}
    path = Path(baseline_bundle) / 'models/text.pt'
    expected_sha = clip['provenance']['baseline']['checkpoint_sha256']['text']
    if sha_file(path) != expected_sha:
        raise ValueError('Text checkpoint differs from the existing experiment')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['run_id'] != base['run_id'] or saved['kind'] != 'text':
        raise ValueError('Wrong text checkpoint')
    head = Head('text'); head.load_state_dict(saved['state_dict']); result['text'] = head.eval().requires_grad_(False)
    for name in ('clip_image', 'clip_fusion'):
        saved = torch.load(Path(clip_bundle) / 'models' / f'{name}.pt', map_location='cpu', weights_only=True)
        if saved['run_id'] != clip['run_id'] or saved['spec'] != clip['heads'][name]:
            raise ValueError('Wrong CLIP checkpoint')
        head = ClipHead(**saved['spec']); head.load_state_dict(saved['state_dict']); result[name] = head.eval().requires_grad_(False)
    return result


def verify_saved_predictions(data, heads, clip_bundle, split, device):
    expected = pd.read_csv(Path(clip_bundle) / 'results' / f'{split}_predictions.csv', dtype={'id': str}).set_index('id')
    if len(expected) != len(data['ids']) or set(expected.index) != set(data['ids']):
        raise ValueError('Saved predictions have different IDs')
    expected = expected.loc[data['ids']]
    if expected.label.tolist() != data['labels'].tolist():
        raise ValueError('Saved prediction labels differ')
    result = {}
    for name, model in heads.items():
        fake, _ = forward_batches(model, data, device)
        column = ('distilbert' if name == 'text' else name) + '_fake_probability'
        if not np.allclose(fake, expected[column], atol=.002, rtol=.002):
            raise ValueError(f'{split}/{name} cannot reproduce saved probabilities; do not train on these caches')
        result[name] = fake
    return result


def train_seed(data, run, config, seed, run_id):
    device = gpu()
    if config['epochs'] < 1 or config['patience'] < 1 or config['batch_size'] < 1:
        raise ValueError('Epochs, patience and batch size must be positive')
    spec = dict(text_dim=data['train']['text'].shape[1], image_dim=data['train']['image'].shape[1], branch_dropout=.2, head_dropout=.3)
    contract = {'run_id': run_id, 'seed': seed, 'config': config, 'spec': spec}
    trial_id = digest(contract)
    folder = Path(run) / 'seeds' / str(seed)
    recovery = folder / 'recovery.pt'
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model = GatedFusion(**spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    start, stale, best_score, best_loss, best_state, history = 0, 0, -1., float('inf'), None, []
    if recovery.exists():
        saved = torch.load(recovery, map_location='cpu', weights_only=True)
        if saved['trial_id'] != trial_id:
            raise ValueError('Recovery configuration changed; use its distinct run directory')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        start, stale, best_score, best_loss = saved['epoch'] + 1, saved['stale'], saved['best_score'], saved['best_loss']
        best_state, history = saved['best_state'], saved['history']
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    train = {k: data['train'][k].to(device) for k in ('text', 'image', 'labels')}
    val = {k: data['validate'][k].to(device) for k in ('text', 'image', 'labels')}
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start, config['epochs']):
        if stale >= config['patience']:
            break
        model.train(); order = torch.randperm(len(train['labels']), device=device); total = 0.
        for ids in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(train['text'][ids], train['image'][ids])
            loss = criterion(logits, train['labels'][ids])  # Raw logits, no softmax before loss.
            loss.backward(); optimizer.step(); total += loss.item() * len(ids)
        model.eval()
        with torch.inference_mode():
            logits, gt, gi = model(val['text'], val['image'])
            val_loss = criterion(logits, val['labels']).item()
            m = metrics(val['labels'].cpu().numpy(), logits.softmax(1)[:, 0].cpu().numpy())
            gates = torch.stack([gt, gi], 1).cpu().numpy()
        if not np.isfinite(val_loss):
            raise ValueError('Non-finite validation loss')
        score = m['macro_f1']
        if (score, -val_loss) > (best_score, -best_loss):
            best_score, best_loss, stale = score, val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({'epoch': epoch + 1, 'train_loss': total / len(order), 'validation_loss': val_loss,
                        'validation_metrics': m, 'validation_gate_statistics': gate_statistics(gates)})
        atomic_save({'trial_id': trial_id, 'epoch': epoch, 'stale': stale, 'best_score': best_score, 'best_loss': best_loss,
                     'best_state': best_state, 'history': history, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, recovery)
        print(f'seed={seed} epoch={epoch + 1} validation macro-F1={score:.4f}')
    if best_state is None:
        raise ValueError('No completed epoch')
    saved = {**contract, 'trial_id': trial_id, 'labels': {'0': 'Fake', '1': 'Real'}, 'state_dict': best_state,
             'validation_macro_f1': best_score, 'validation_loss': best_loss}
    atomic_save(saved, folder / 'best.pt'); save_json(history, folder / 'history.json')
    return saved


def select_seeds(data, run, config, seeds, run_id):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Supply distinct random seeds')
    records = [train_seed({s: data[s] for s in ('train', 'validate')}, run, config, seed, run_id) for seed in seeds]
    chosen = max(records, key=lambda r: (r['validation_macro_f1'], -r['validation_loss']))
    atomic_save(chosen, Path(run) / 'gated_fusion.pt')
    values = [r['validation_macro_f1'] for r in records]
    selection = {'criterion': 'Highest validation macro-F1, then lowest validation loss, then seed list order. No test or Fargo selection.',
                 'selected_seed': chosen['seed'], 'selected_trial_id': chosen['trial_id'],
                 'validation_macro_f1_mean': float(np.mean(values)), 'validation_macro_f1_std': float(np.std(values)),
                 'seeds': [{k: r[k] for k in ('seed', 'validation_macro_f1', 'validation_loss')} for r in records]}
    save_json(selection, Path(run) / 'selection.json')
    model = GatedFusion(**chosen['spec']); model.load_state_dict(chosen['state_dict'])
    return model.eval(), selection


def evaluate_selected(model, data, original, threshold, run, split, device):
    run = Path(run); output = run / 'results' / split; output.mkdir(parents=True, exist_ok=True)
    fake, gates = forward_batches(model, data, device)
    disagreement = disagreement_report(data['labels'].numpy(), original['text'], original['clip_image'], fake, threshold, split)
    report = {'split': split, 'gated': metrics(data['labels'].numpy(), fake),
              'concatenation': metrics(data['labels'].numpy(), original['clip_fusion']),
              'gates': gate_statistics(gates), 'disagreement': disagreement}
    save_json(report, output / 'metrics.json')
    pd.DataFrame({'id': data['ids'], 'label': data['labels'].tolist(), 'text_fake': original['text'],
                  'clip_image_fake': original['clip_image'], 'concatenation_fake': original['clip_fusion'],
                  'gated_fake': fake, 'g_text': gates[:, 0], 'g_image': gates[:, 1],
                  'disagreement': np.abs(original['text'] - original['clip_image'])}).to_csv(output / 'predictions.csv', index=False)
    selection = json.loads((run / 'selection.json').read_text())
    history = json.loads((run / 'seeds' / str(selection['selected_seed']) / 'history.json').read_text())
    diagnostic_plots(data['labels'].numpy(), {'Concatenation': original['clip_fusion'], 'Gated': fake}, gates, history, disagreement, output)
    if split == 'validate':
        save_json(warning_policy(disagreement), run / 'warning_policy.json')
    save_json({'complete': True, 'split': split}, output / 'complete.json')
    return report
