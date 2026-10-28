"""Build the isolated, self-contained CLIP experiment notebook (never edits baseline notebook)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = ("dataset.py", "models.py", "pipeline.py", "clip_models.py", "clip_experiment.py",
         "clip_inference.py", "clip_requirements.txt", "CLIP_EXPERIMENT.md")


def build():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)})

    md("""# Frozen CLIP Vision experiment — separate from the working prototype
Use a **new Colab GPU runtime**. Run cells sequentially. Existing baseline checkpoints, feature caches, processed data and Streamlit files are read-only.
The experiment adds CLIP image-only and cached DistilBERT + CLIP fusion. It never downloads Fakeddit, re-splits data, fine-tunes encoders or modifies the app.
CLIP's projected image features are **512-dimensional**; cached DistilBERT is **768-dimensional**. All classification heads encode their actual input dimensions.
References: [CLIP API](https://huggingface.co/docs/transformers/model_doc/clip), [model card](https://huggingface.co/openai/clip-vit-base-patch32).
""")
    md("## 1. Dependencies and GPU check\nKeep Colab's matched torch/torchvision CUDA packages. Only Python dependencies and pretrained CLIP weights may be downloaded.")
    code('''%pip -q install "transformers==4.57.1" "pandas>=2.2,<3" "scikit-learn>=1.5,<2" "matplotlib>=3.8,<4" "Pillow>=10,<13" safetensors
import torch, torchvision
assert torch.cuda.is_available(), "Runtime > Change runtime type > GPU is required."
print(torch.cuda.get_device_name(0), torch.__version__, torchvision.__version__)
''')
    md("## 2. Mount Drive and install embedded source into a separate temporary directory")
    sources = {name: (ROOT / name).read_text(encoding="utf-8") for name in FILES}
    code('''from google.colab import drive, files
from pathlib import Path
import os, sys, json, shutil, gc, hashlib
drive.mount('/content/drive')
WORK = Path('/content/fakeddit_clip_code')
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
SOURCE_FILES = ''' + repr(sources) + '''
for name, source in SOURCE_FILES.items():
    (WORK / name).write_text(source, encoding='utf-8')
''')
    md("""## 3. Set paths and validation-only sweep
Use the original Drive outputs from colab_training.ipynb. The known baseline run ID below comes from this project's trained Windows bundle; edit if using a different run.
The original features folder is required for both cached DistilBERT reuse and fair ResNet50 comparison. A Windows export alone does not contain these feature caches.
Set the hyperparameter grid and promotion gain **before** looking at test results. Do not change the working app.
""")
    code('''DATA_SOURCE = "drive"  # "drive" or "zip"
DATA_DIR = Path('/content/drive/MyDrive/Multimodal_Fake_News_Detection/data')
ZIP_ON_DRIVE = ""  # Optional existing processed ZIP; otherwise upload when DATA_SOURCE == "zip".
BASELINE_ROOT = Path('/content/drive/MyDrive/Fakeddit_Demo')
BASELINE_RUN = BASELINE_ROOT / 'runs/27751e2dbfb636fc5929'
BASELINE_CACHE_ROOT = BASELINE_ROOT / 'features'
TEXT_ASSETS = BASELINE_ROOT / 'encoders_v1/distilbert'
OUTPUT = Path('/content/drive/MyDrive/Fakeddit_CLIP_Experiment')
CLIP_REVISION = "main"  # Resolved once to an immutable commit and saved for recovery.
FEATURE_BATCH_SIZE = 32  # Can reduce without losing completed fixed-size feature shards.
TRIAL_CONFIGS = [dict(seed=42, epochs=25, batch_size=64, lr=lr, weight_decay=0.0001,
                      patience=5, dropout=0.3) for lr in (0.001, 0.0003)]
MINIMUM_VALIDATION_GAIN = 0.0  # Strict macro-F1 gain above the corresponding baseline.
from dataset import extract_zip, validate
from clip_experiment import guard_output
guard_output(OUTPUT, BASELINE_ROOT, BASELINE_RUN, BASELINE_CACHE_ROOT, TEXT_ASSETS, DATA_DIR)
OUTPUT.mkdir(parents=True, exist_ok=True)
if DATA_SOURCE == 'zip':
    if ZIP_ON_DRIVE:
        archive = Path(ZIP_ON_DRIVE)
    else:
        uploaded = files.upload()
        assert len(uploaded) == 1, 'Upload exactly one processed dataset ZIP.'
        archive = WORK / next(iter(uploaded))
        del uploaded
    DATA_DIR = extract_zip(archive, Path('/content/fakeddit_clip_data'))
elif DATA_SOURCE != 'drive':
    raise ValueError('Choose drive or zip')
guard_output(OUTPUT, DATA_DIR)
assert (BASELINE_RUN / 'config.json').is_file(), f'Choose the existing baseline run: {BASELINE_RUN}'
assert (TEXT_ASSETS / 'config.json').is_file(), f'Missing original DistilBERT assets: {TEXT_ASSETS}'
''')
    md("""## 4. Validate unchanged data and load existing caches by sample ID
Checks all 3,000/500/500 records/images, labels (0=Fake, 1=Real), duplicates and overlaps. No rows are dropped. Windows/Linux image paths are resolved without editing CSVs.
Baseline fingerprints must match the dataset. Cached text and ResNet50 embeddings are joined by ID, not by file order. Missing/inconsistent caches stop the experiment.
""")
    code('''from clip_experiment import (read_baseline_config, load_baseline_features, cache_hashes,
    prepare_clip, extract_clip, paired_features, digest, sha_file, select_trials,
    compare_models, promotion_gate, plot_comparison, export_bundle)
from pipeline import fingerprint, write_json
from clip_models import FEATURE_SPEC, CachedTextEncoder
frames, validation_report = validate(DATA_DIR, OUTPUT / 'dataset_validation.json')
baseline_config, baseline_provenance = read_baseline_config(BASELINE_RUN, frames)
baseline = load_baseline_features(BASELINE_CACHE_ROOT, baseline_config['feature_id'], frames)
baseline_provenance['cache_sha256'] = cache_hashes(baseline)
baseline_provenance['text_asset_sha256'] = {p.relative_to(TEXT_ASSETS).as_posix(): sha_file(p)
    for p in TEXT_ASSETS.rglob('*') if p.is_file()}
print({s: len(d['ids']) for s, d in baseline.items()})
# Verify that exported text encoder assets actually reproduce the cached text representation.
text_encoder = CachedTextEncoder(TEXT_ASSETS, device='cuda')
with torch.inference_mode():
    actual = text_encoder.encode(frames['validate'].clean_title.iloc[:2].tolist()).cpu()
torch.testing.assert_close(actual, baseline['validate']['text'][:2], atol=2e-3, rtol=2e-3)
del text_encoder
gc.collect()
torch.cuda.empty_cache()
print('Original data, labels, caches and text encoder verified.')
''')
    md("""## 5. Load frozen CLIP Vision and extract resumable image embeddings
Uses CLIPVisionModelWithProjection.image_embeds, 512-D, L2-normalized. The saved CLIPImageProcessor applies model-required transforms to existing RGB images. Baseline preprocessing is unchanged.
All encoder parameters, including the pretrained projection, are frozen. Completed 128-example shards survive runtime disconnections; only an incomplete shard is repeated.
""")
    code('''clip_encoder, CLIP_ASSETS, clip_manifest = prepare_clip(OUTPUT, CLIP_REVISION)
assert not clip_encoder.model.training
assert all(not p.requires_grad for p in clip_encoder.model.parameters())
assert clip_encoder.dimension == 512
feature_contract = {'feature_spec': FEATURE_SPEC, 'clip_assets': clip_manifest,
                    'encoder_code_sha256': hashlib.sha256(SOURCE_FILES['clip_models.py'].encode()).hexdigest()}
CLIP_FEATURE_ID = fingerprint(frames, feature_contract)
visual = extract_clip(frames, clip_encoder, OUTPUT / 'features', CLIP_FEATURE_ID, FEATURE_BATCH_SIZE)
clip_data = paired_features(baseline, visual)
for split, frame in frames.items():
    assert clip_data[split]['image'].shape == (len(frame), 512)
    assert clip_data[split]['text'].shape == (len(frame), 768)
    assert clip_data[split]['ids'] == frame.id.tolist()
del clip_encoder
gc.collect()
torch.cuda.empty_cache()
run_contract = {'version': 1, 'clip_feature_id': CLIP_FEATURE_ID, 'baseline': baseline_provenance,
                'trials': TRIAL_CONFIGS, 'minimum_validation_gain': MINIMUM_VALIDATION_GAIN,
                'source_sha256': {name: hashlib.sha256(source.encode()).hexdigest()
                                  for name, source in SOURCE_FILES.items() if name.endswith('.py')}}
RUN_ID = digest(run_contract)
RUN = OUTPUT / 'runs' / RUN_ID
RUN.mkdir(parents=True, exist_ok=True)
write_json(validation_report, RUN / 'results/dataset_validation.json')
write_json(run_contract, RUN / 'provenance.json')
print('New isolated experiment run:', RUN)
''')
    md("""## 6. Train image-only and fusion heads; select on validation macro-F1
Only train tensors update weights. Validation macro-F1 selects checkpoints/trials (loss breaks ties).
This cell passes **no test data** to training. Resume restores last/best weights, AdamW state, RNG and patience. Completed trials are reused.
""")
    code('''training_data = {s: clip_data[s] for s in ('train', 'validate')}
selected = select_trials(training_data, RUN, TRIAL_CONFIGS, RUN_ID)
config = {'run_id': RUN_ID, 'feature_spec': FEATURE_SPEC, 'labels': {'0': 'Fake', '1': 'Real'},
          'heads': {kind: info['spec'] for kind, info in selected.items()},
          'clip_revision': clip_manifest['resolved_revision'], 'clip_feature_id': CLIP_FEATURE_ID,
          'baseline_run_id': baseline_config['run_id'], 'text_encoder': baseline_config['encoders'],
          'selection_metric': 'validation macro-F1', 'provenance': run_contract}
write_json(config, RUN / 'config.json')
print(json.dumps(selected, indent=2))
''')
    md("""## 7. Compare validation scores and lock the integration decision
Re-evaluates the original three checkpoints without retraining them, adding macro-F1. CLIP image-only is compared to ResNet50; CLIP fusion to DistilBERT + ResNet50.
Eligibility requires strict validation macro-F1 improvement above the configured minimum. No claim of statistical significance. The test set cannot influence this decision.
""")
    code('''validation = compare_models(baseline, clip_data, BASELINE_RUN, RUN, RUN_ID, splits=('validate',))['validate']
gate = promotion_gate(validation, MINIMUM_VALIDATION_GAIN)
write_json(gate, RUN / 'results/promotion.json')
decision_lines = ['# Validation-based integration decision', '', gate['criterion'], '']
for kind, decision in gate['candidates'].items():
    status = 'Eligible for an optional integration trial' if decision['eligible'] else 'Not eligible; keep the existing model'
    decision_lines.append(f"- {kind}: {status}. Validation macro-F1 change: {decision['delta_macro_f1']:+.6f}.")
decision_lines += ['', 'See CLIP_EXPERIMENT.md for conditional integration steps. No Streamlit app has been modified.',
                   'Test results must not be used to change this validation-based selection.']
(RUN / 'results/INTEGRATION_DECISION.md').write_text('\\n'.join(decision_lines), encoding='utf-8')
plot_comparison(RUN, 'validate')
import pandas as pd
from IPython.display import display, Image as DisplayImage
display(pd.DataFrame(validation['models']).T)
display(DisplayImage(filename=str(RUN / 'results/validate_comparison.png')))
print(json.dumps(gate, indent=2))
''')
    md("""## 8. Final comparison on the same 500 test examples
All heads are already selected. Reports accuracy, Fake-positive precision/recall/F1, macro-F1, ROC-AUC and confusion matrices (Fake, Real).
Always reports test results, even when validation did not improve. Do not tune against these results.
""")
    code('''assert (RUN / 'selection.json').exists() and (RUN / 'results/promotion.json').exists()
test = compare_models(baseline, clip_data, BASELINE_RUN, RUN, RUN_ID, splits=('test',))['test']
plot_comparison(RUN, 'test')
write_json({'validation': validation, 'test': test, 'promotion': gate}, RUN / 'results/metrics.json')
display(pd.DataFrame(test['models']).T)
display(DisplayImage(filename=str(RUN / 'results/test_comparison.png')))
display(DisplayImage(filename=str(RUN / 'results/test_confusion_matrices.png')))
''')
    md("""## 9. Build a separate CPU export and verify prediction parity
Exports frozen CLIP + its processor, original DistilBERT, selected experimental heads/configs/results, and an offline CPU loader. Contains no app.py or production artifacts folder.
CPU predictions must agree with cached GPU-feature predictions before the ZIP is published. This is a single-pair compatibility check, not CPU training.
""")
    code('''from clip_inference import ClipPredictor
from PIL import Image
import importlib.metadata
import zipfile
EXPORT_DIR = Path('/content/clip_exports') / RUN_ID
BUNDLE = export_bundle(RUN, CLIP_ASSETS, TEXT_ASSETS, EXPORT_DIR, config, WORK)
shutil.copy2(RUN / 'provenance.json', BUNDLE / 'provenance.json')
versions = {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'transformers', 'numpy', 'pandas', 'Pillow', 'scikit-learn')}
write_json(versions, BUNDLE / 'environment.json')
torch.set_num_threads(2)
predictor = ClipPredictor(BUNDLE)
row = frames['test'].iloc[0]
with Image.open(row.resolved_path) as image:
    predictions = predictor.predict(row.clean_title, image)
for kind, head in predictor.heads.items():
    with torch.inference_mode():
        expected = head(clip_data['test']['text'][:1], clip_data['test']['image'][:1]).softmax(1)[0]
    torch.testing.assert_close(torch.tensor(predictions[kind]), expected, atol=2e-3, rtol=2e-3)
smoke = {'passed': True, 'sample_id': row.id, 'probabilities': predictions,
         'check': 'offline CPU vs cached GPU features; atol=rtol=0.002'}
write_json(smoke, BUNDLE / 'results/cpu_smoke.json')
write_json(smoke, RUN / 'results/cpu_smoke.json')
# Check that no baseline checkpoint changed during the experiment.
for kind, expected_hash in baseline_provenance['checkpoint_sha256'].items():
    assert sha_file(BASELINE_RUN / 'models' / f'{kind}.pt') == expected_hash
manifest = {p.relative_to(EXPORT_DIR).as_posix(): sha_file(p) for p in EXPORT_DIR.rglob('*')
            if p.is_file() and p.name != 'SHA256SUMS.json'}
write_json(manifest, EXPORT_DIR / 'SHA256SUMS.json')
export_path = OUTPUT / f'clip_windows_{RUN_ID}.zip'
temporary = export_path.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in EXPORT_DIR.rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(EXPORT_DIR).as_posix())
os.replace(temporary, export_path)
print('CPU check passed. Download the separate bundle from Drive:', export_path)
print('The working Streamlit app and baseline checkpoints remain unchanged.')
''')
    md("""## 10. Download and review
Prefer downloading the ZIP from Google Drive. Extract into a new Windows folder; follow CLIP_EXPERIMENT.md to run CPU inference.
Only models marked eligible in results/promotion.json should be considered for optional integration into a copy of the app. No integration happens automatically.
After a disconnect, rerun cells sequentially with identical settings; persistent shards and trial recovery are reused.
""")
    code("# Optional browser download (the ZIP is already saved in Drive):\n# files.download(str(export_path))\n")
    for index, cell in enumerate(cells):
        cell["id"] = f"clip-{index:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "colab": {"name": "colab_clip_experiment.ipynb"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
    (ROOT / "colab_clip_experiment.ipynb").write_text(json.dumps(notebook, indent=1), encoding="utf-8")


if __name__ == "__main__":
    build()
    print("Wrote colab_clip_experiment.ipynb")
