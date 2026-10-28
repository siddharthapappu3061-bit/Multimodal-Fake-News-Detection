"""Export separate real weights and verify CPU predictions before declaring deployment ready."""
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from clip_experiment import sha_file, guard_output
from gated_analysis import save_json, forward_batches


def export_bundle(run, baseline, clip, text_assets, image_assets, source_dir):
    run, baseline, clip = map(Path, (run, baseline, clip))
    destination = run / 'windows_export'
    guard_output(destination, baseline, clip, text_assets, image_assets)
    config = json.loads((clip / 'config.json').read_text())
    # The original tokenizer and text weights must be identical, not merely the same model name.
    for name, expected in config['provenance']['baseline']['text_asset_sha256'].items():
        path = Path(text_assets) / name
        if not path.is_file() or sha_file(path) != expected:
            raise ValueError(f'Missing/different original DistilBERT asset: {path}')
    for folder, filenames in [(image_assets, ['config.json', 'preprocessor_config.json']), (text_assets, ['config.json'])]:
        for name in filenames:
            if not (Path(folder) / name).is_file():
                raise FileNotFoundError(Path(folder) / name)
    bundle = destination / 'gated_artifacts'
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / 'models').mkdir(exist_ok=True)
    shutil.copy2(run / 'gated_fusion.pt', bundle / 'gated_fusion.pt')
    shutil.copy2(baseline / 'models/text.pt', bundle / 'models/text.pt')
    for name in ('clip_image', 'clip_fusion'):
        shutil.copy2(clip / 'models' / (name + '.pt'), bundle / 'models' / (name + '.pt'))
    for kind, folder in [('distilbert', text_assets), ('clip', image_assets)]:
        shutil.copytree(folder, bundle / 'encoders' / kind, dirs_exist_ok=True)
    checkpoint = torch.load(run / 'gated_fusion.pt', map_location='cpu', weights_only=True)
    export_config = {k: checkpoint[k] for k in ['run_id', 'spec', 'seed', 'labels', 'config']}
    export_config.update(feature_spec=config['feature_spec'], text_encoder=config['text_encoder'],
                         warning_policy=json.loads((run / 'warning_policy.json').read_text()),
                         baseline_run_id=config['baseline_run_id'], clip_run_id=config['run_id'])
    save_json(export_config, bundle / 'config.json')
    for name in ('selection.json', 'experiment_config.json'):
        shutil.copy2(run / name, bundle / name)
    shutil.copytree(run / 'results', bundle / 'results', dirs_exist_ok=True)
    for name in ('gated_inference.py', 'gated_model.py', 'clip_models.py', 'models.py'):
        shutil.copy2(Path(source_dir) / name, destination / name)
    (destination / 'requirements.txt').write_text('torch\ntorchvision\ntransformers==4.57.1\nPillow\nsafetensors\n', encoding='utf-8')
    save_json({p.relative_to(bundle).as_posix(): sha_file(p) for p in bundle.rglob('*') if p.is_file() and p.name != 'sha256.json'}, bundle / 'sha256.json')
    return destination


def cpu_smoke(export_dir, frame, cached, model, output):
    from gated_inference import GatedPredictor
    start = time.perf_counter()
    predictor = GatedPredictor(Path(export_dir) / 'gated_artifacts')
    cold_load = time.perf_counter() - start
    rows = []
    # First three validation samples are predeclared; no cherry-picking or test selection.
    for index in range(min(3, len(frame))):
        row = frame.iloc[index]
        inputs = {k: cached[k][index:index+1] for k in ('text', 'image')}
        expected, gates = forward_batches(model, inputs, 'cpu')
        predictions, seconds = [], []
        with Image.open(row.resolved_path) as image:
            with torch.inference_mode():
                raw_text = predictor.text.encode([row.clean_title])
                raw_image = predictor.image.encode([image])
            for kind, actual_feature in [('text', raw_text), ('image', raw_image)]:
                if not torch.allclose(actual_feature, inputs[kind], atol=.002, rtol=.002):
                    raise ValueError(f'Exported {kind} encoder does not reproduce original cached features')
            for repeat in range(2):
                start = time.perf_counter(); predictions.append(predictor.predict(row.clean_title, image)); seconds.append(time.perf_counter() - start)
        actual = predictions[0]
        with torch.inference_mode():
            for name, head in predictor.heads.items():
                expected_reference = head(inputs['text'], inputs['image']).softmax(1)[0].numpy()
                if not np.allclose(actual['probabilities'][name], expected_reference, atol=.002, rtol=.002):
                    raise ValueError(f'Exported {name} baseline parity failed')
        if actual != predictions[1]:
            raise ValueError('Repeated CPU inference is inconsistent')
        if not np.allclose(actual['probabilities']['gated_fusion'][0], expected[0], atol=.002, rtol=.002):
            raise ValueError('Raw-input CPU inference differs from cached-feature predictions')
        if not np.allclose([actual['g_text'], actual['g_image']], gates[0], atol=.002, rtol=.002):
            raise ValueError('CPU gate mismatch')
        rows.append({'id': row.id, 'prediction': actual, 'cached_fake': float(expected[0]), 'seconds': seconds})
    save_json({'passed': True, 'cold_load_seconds': cold_load, 'samples': rows}, output)
    return predictor
