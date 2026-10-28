"""Read-only baseline inspection and genuine saved-prediction diagnostics; no training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl-cache'))

import numpy as np
import pandas as pd
from PIL import Image
import torch
from dataset import resolve_image
from integrated_inference import FiveModelPredictor
from gated_analysis import choose_threshold, disagreement_report, save_json, plotting
from gated_pipeline import FARGO_TITLE


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    output = HERE / 'local_baseline'; output.mkdir(exist_ok=True)
    base, clip = ROOT / 'windows_demo/artifacts', ROOT / 'CLIP-vision/clip_artifacts'
    base_config, clip_config = [json.loads((p / 'config.json').read_text()) for p in (base, clip)]
    source_files = [*ROOT.glob('*.py'), *ROOT.glob('*.ipynb'), *ROOT.glob('*.md'), ROOT / 'windows_demo/app.py']
    protected = source_files + [p for parent in (base, clip, ROOT / 'data') for p in parent.rglob('*') if p.is_file()]
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected}
    save_json(manifest, output / 'protected_sha256.json')
    frames = {split: pd.read_csv(ROOT / 'data' / f'{split}_processed.csv', dtype={'id': str}) for split in ('train', 'validate', 'test')}
    data = {}
    for split in ('validate', 'test'):
        df = pd.read_csv(clip / 'results' / f'{split}_predictions.csv', dtype={'id': str})
        if df.id.tolist() != frames[split].id.tolist() or df.label.tolist() != frames[split]['2_way_label'].tolist():
            raise ValueError('Saved predictions differ from original split')
        data[split] = df
    threshold = choose_threshold(data['validate'].distilbert_fake_probability, data['validate'].clip_image_fake_probability)
    save_json(threshold, output / 'baseline_threshold.json')
    for split, df in data.items():
        report = disagreement_report(df.label, df.distilbert_fake_probability, df.clip_image_fake_probability,
                                     df.clip_fusion_fake_probability, threshold, split)
        report['model'] = 'existing DistilBERT + CLIP concatenation; NOT a gated model'
        save_json(report, output / f'{split}_disagreement.json')
        plt = plotting()
        bins = report['bins']
        plt.bar([f"{b['lower']:.1f}–{b['upper']:.1f}" for b in bins], [b['error_rate'] or 0 for b in bins])
        plt.title('Existing concatenation fusion: ' + split); plt.xlabel('Text/image disagreement'); plt.ylabel('Error rate')
        plt.tight_layout(); plt.savefig(output / f'{split}_disagreement.png'); plt.close()
    torch.set_num_threads(2)
    start = time.perf_counter(); predictor = FiveModelPredictor(base, clip); cold = time.perf_counter() - start
    reproduced = {}
    for sample_id in ('bdlmf3', 'ba8vze'):
        row = frames['test'].set_index('id').loc[sample_id]
        with Image.open(resolve_image(ROOT / 'data', row.local_image_path)) as image:
            start = time.perf_counter(); actual = predictor.predict(row.clean_title, image); elapsed = time.perf_counter() - start
        expected = data['test'].set_index('id').loc[sample_id]
        for name in ('clip_image', 'clip_fusion'):
            if not np.isclose(actual[name][0], expected[name + '_fake_probability'], atol=.002, rtol=.002):
                raise ValueError('Existing CLIP predictions did not reproduce')
        reproduced[sample_id] = {'probabilities': actual, 'inference_seconds': elapsed}
    save_json({'passed': True, 'samples': reproduced, 'cold_load_seconds': cold,
               'device': 'Windows CPU; existing five-model pipeline only'}, output / 'baseline_reproduction.json')
    fargo_source = Path.home() / 'Downloads/flamethrower.jpg'
    if not fargo_source.is_file():
        save_json({'status': 'blocked', 'missing': str(fargo_source)}, output / 'fargo.json')
    else:
        target = HERE / 'diagnostics/fargo.jpg'; target.parent.mkdir(exist_ok=True)
        if target.exists() and sha(target) != sha(fargo_source):
            raise ValueError('Different diagnostic already preserved; do not replace it silently')
        if not target.exists():
            shutil.copy2(fargo_source, target)
        normalized = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
        if any(normalized(t) == normalized(FARGO_TITLE) for df in frames.values() for t in df.clean_title):
            raise ValueError('Fargo appears in original dataset; stop without modifying splits')
        fargo_sha = sha(target)
        if fargo_sha in [v for p, v in manifest.items() if p.startswith('data/images/')]:
            raise ValueError('Fargo diagnostic image is already in dataset')
        with Image.open(target) as image:
            actual = predictor.predict(FARGO_TITLE, image)
        save_json({'headline': FARGO_TITLE, 'image_sha256': fargo_sha, 'label': 0, 'label_source': 'User supplied Fake label',
                   'usage': 'Qualitative only; excluded from training, validation selection, and tuning',
                   'measured_probabilities': actual,
                   'previous_user_reported_fake_probabilities': {'text': .004, 'image': .75, 'clip_image': .907, 'fusion': .163, 'clip_fusion': .06},
                   'disagreement': abs(actual['text'][0] - actual['clip_image'][0]),
                   'gated_probability': None, 'gates': None, 'gated_status': 'Pending Colab training and export'}, output / 'fargo.json')
    inspection = {'baseline_run_id': base_config['run_id'], 'clip_run_id': clip_config['run_id'],
                  'text_feature_id': base_config['feature_id'], 'clip_feature_id': clip_config['clip_feature_id'],
                  'checkpoint_dimensions': clip_config['heads']['clip_fusion'], 'labels': base_config['labels'],
                  'local_embedding_caches': 'Missing: no original text/CLIP feature shards in the local project',
                  'required_drive_paths': [f"Fakeddit_Demo/features/{base_config['feature_id']}/{{train,validate,test}}/text/*.pt",
                                           f"Fakeddit_CLIP_Experiment/features/{clip_config['clip_feature_id']}/{{train,validate,test}}/*.pt"],
                  'ablation_status': 'Pending verified Drive cache access; not inferred from disagreement alone',
                  'gated_training_status': 'Pending Colab GPU; no gated checkpoint or performance claim',
                  'baseline_reproduction': 'passed', 'streamlit': 'Unmodified; sixth-model integration deferred'}
    save_json(inspection, output / 'inspection.json')
    for name, expected in manifest.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Protected file changed: ' + name)
    print(json.dumps(inspection, indent=2))


if __name__ == '__main__':
    main()
