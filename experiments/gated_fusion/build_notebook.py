"""Build the self-contained isolated Colab experiment. No original files are changed."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build():
    cells = []
    def md(text):
        cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))
    def code(text):
        cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))
    md('''# Gated DistilBERT + CLIP fusion — independent experiment
Run sequentially in a **Colab GPU runtime**. Reuse the existing 3,000/500/500 pairs and feature shards. No dataset downloads, encoder training, cache extraction or Streamlit changes occur here.
Labels: **0 = Fake, 1 = Real**. Validation macro-F1 selects the checkpoint and seed. Test evaluation follows selection. Fargo is qualitative only.
Missing or inconsistent source artifacts stop execution; they are never replaced with synthetic data.
''')
    code('''%pip -q install "transformers==4.57.1" "pandas>=2.2,<3" "scikit-learn>=1.5,<2" "matplotlib>=3.8,<4" "Pillow>=10,<13" safetensors
import torch, torchvision
assert torch.cuda.is_available(), 'Choose Runtime > Change runtime type > T4 GPU.'
print(torch.cuda.get_device_name(0), torch.__version__, torchvision.__version__)
''')
    md('## 1. Mount Drive and write isolated embedded source')
    sources = {name: (ROOT / name).read_text(encoding='utf-8') for name in ['dataset.py', 'models.py', 'pipeline.py', 'clip_models.py', 'clip_experiment.py']}
    sources.update({name: (HERE / name).read_text(encoding='utf-8') for name in ['gated_model.py', 'gated_analysis.py', 'gated_pipeline.py', 'gated_inference.py', 'gated_export.py']})
    code('''from google.colab import drive, files
from pathlib import Path
import os, sys, json, hashlib, shutil
drive.mount('/content/drive')
WORK = Path('/content/fakeddit_gated_code')
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
SOURCE_FILES = ''' + repr(sources) + '''
for name, source in SOURCE_FILES.items():
    (WORK / name).write_text(source, encoding='utf-8')
''')
    md('''## 2. Paths and predeclared settings
Point BASELINE and CLIP at the **existing run directories** (or exported bundles with config.json, models/, results/). Cache roots are their original **features** directories, not Windows exports. Uploading a dataset ZIP does not restore missing caches.
For CPU export, set TEXT_ASSETS and CLIP_ASSETS to the corresponding existing encoder folders. A complete existing clip_artifacts export contains both. These are checked at export; head training needs only caches.
Do not change these settings after inspecting test results. The optional Fargo path can be supplied now solely to check exclusion, or uploaded at the final diagnostic cell.
''')
    code('''DRIVE = Path('/content/drive/MyDrive')
BASELINE = DRIVE / 'Fakeddit_Demo/runs/27751e2dbfb636fc5929'
CLIP = DRIVE / 'Fakeddit_CLIP_Experiment/runs/29037a54744b05c6dc32'
TEXT_CACHE = DRIVE / 'Fakeddit_Demo/features'
CLIP_CACHE = DRIVE / 'Fakeddit_CLIP_Experiment/features'
DATA_DIR = DRIVE / 'Multimodal_Fake_News_Detection/data'
DATA_SOURCE = 'drive'  # 'drive' or 'zip'
ZIP_ON_DRIVE = ''  # Existing processed dataset ZIP; blank prompts upload in zip mode.
TEXT_ASSETS = DRIVE / 'Fakeddit_Demo/encoders_v1/distilbert'
# Set this to the original encoder folder or an uploaded clip_artifacts/encoders/clip.
CLIP_ASSETS = DRIVE / 'clip_artifacts/encoders/clip'
FARGO_IMAGE = ''
OUTPUT = DRIVE / 'Fakeddit_Gated_Fusion'
TRAINING = dict(lr=.0003, batch_size=64, epochs=30, weight_decay=.001, patience=5)
SEEDS = [42, 43, 44]  # Choose before training. Keep unchanged when resuming.
DISAGREEMENT_QUANTILE = .8
FIXED_DISAGREEMENT_THRESHOLD = None  # Optional predeclared threshold in [0,1].
ABLATION_SEED, ABLATION_REPETITIONS = 42, 5
from dataset import validate, extract_zip
from clip_experiment import guard_output, digest, sha_file
from gated_analysis import *
from gated_pipeline import *
guard_output(OUTPUT, BASELINE, CLIP, TEXT_CACHE, CLIP_CACHE, DATA_DIR, TEXT_ASSETS, CLIP_ASSETS)
OUTPUT.mkdir(parents=True, exist_ok=True)
if DATA_SOURCE == 'zip':
    if ZIP_ON_DRIVE:
        archive = Path(ZIP_ON_DRIVE)
    else:
        uploaded = files.upload()
        assert len(uploaded) == 1, 'Upload exactly one existing processed dataset ZIP.'
        archive = WORK / next(iter(uploaded))
        del uploaded
    DATA_DIR = extract_zip(archive, Path('/content/fakeddit_gated_data'))
elif DATA_SOURCE != 'drive':
    raise ValueError('DATA_SOURCE must be drive or zip')
guard_output(OUTPUT, DATA_DIR)
for folder in (BASELINE, CLIP):
    assert (folder / 'config.json').is_file(), f'Missing existing run configuration: {folder}/config.json'
''')
    md('''## 3. Verify original data, labels, sample alignment and caches
Validate every image, preserve CSVs, resolve Windows paths on Linux, join feature shards by ID, verify labels and text cache hashes. CLIP feature IDs and measured dimensions must match the original trained heads. No rows are dropped.
''')
    code('''frames, dataset_report = validate(DATA_DIR, OUTPUT / 'dataset_validation.json')
data, base_config, clip_config, dimensions = load_caches(frames, BASELINE, CLIP, TEXT_CACHE, CLIP_CACHE, FARGO_IMAGE or None)
heads = existing_heads(BASELINE, CLIP)
cache_hashes = {s: {k: hashlib.sha256(d[k].contiguous().numpy().tobytes()).hexdigest() for k in ('text','image')} for s,d in data.items()}
contract = dict(training=TRAINING, seeds=SEEDS, dimensions=dimensions, labels=base_config['labels'],
    baseline_run_id=base_config['run_id'], clip_run_id=clip_config['run_id'], cache_hashes=cache_hashes,
    text_encoder=base_config['encoders'], feature_spec=clip_config['feature_spec'],
    source_sha256={name: hashlib.sha256(source.encode()).hexdigest() for name,source in SOURCE_FILES.items()},
    threshold_quantile=DISAGREEMENT_QUANTILE, fixed_threshold=FIXED_DISAGREEMENT_THRESHOLD,
    ablation_seed=ABLATION_SEED, ablation_repetitions=ABLATION_REPETITIONS)
RUN_ID = digest(contract)
RUN = OUTPUT / 'runs' / RUN_ID
RUN.mkdir(parents=True, exist_ok=True)
save_json(contract, RUN / 'experiment_config.json')
original_validation = verify_saved_predictions(data['validate'], heads, CLIP, 'validate', 'cuda')
print('Verified:', dimensions, {s:len(d['ids']) for s,d in data.items()}, RUN)
''')
    md('''## 4. Original concatenation ablations — before creating/training the gated model
Five label-blind derangements, training-mean replacements and Gaussian perturbations. Save per-ID changes, class flips, plots and a descriptive bootstrap interval. Intervention accuracy is omitted: mismatched pairs do not have justified labels. These diagnostics cannot prove causal importance or that a modality is ignored.
''')
    code('''baseline_ablation = ablations(heads['clip_fusion'], data['validate'], data['train'], RUN / 'ablations/concatenation_validate',
    'cuda', seed=ABLATION_SEED, repetitions=ABLATION_REPETITIONS)
print(json.dumps(baseline_ablation, indent=2))
threshold = choose_threshold(original_validation['text'], original_validation['clip_image'],
    quantile=DISAGREEMENT_QUANTILE, fixed=FIXED_DISAGREEMENT_THRESHOLD)
save_json(threshold, RUN / 'threshold.json')
''')
    md('''## 5. Train only the gated head; recover completed epochs and seeds
Each modality projects to 256 with LayerNorm, ReLU and dropout .2. A 512→128→2 softmax gate weights the branches. Their concatenation feeds a 512→128→2 head with dropout .3. Cross-entropy receives raw logits. Encoders are never involved in training.
AdamW, validation macro-F1 selection, patience 5. Save optimizer, CPU/CUDA RNG, history, best weights every epoch. Rerun from the top after disconnecting with unchanged settings: completed epochs/seeds are recovered. No feature extraction repeats because this notebook never extracts caches.
''')
    code('''assert (RUN / 'ablations/concatenation_validate/report.json').is_file()
gated, selection = select_seeds(data, RUN, TRAINING, SEEDS, RUN_ID)
validation = evaluate_selected(gated, data['validate'], original_validation, threshold, RUN, 'validate', 'cuda')
gated_ablation = ablations(gated, data['validate'], data['train'], RUN / 'ablations/gated_validate',
    'cuda', seed=ABLATION_SEED, repetitions=ABLATION_REPETITIONS)
improved = validation['gated']['macro_f1'] > validation['concatenation']['macro_f1']
save_json(dict(validation_improved=improved, selection=selection, threshold=threshold,
    decision='Frozen before held-out evaluation; no automatic app promotion'), RUN / 'final_selection.json')
print('Validation macro-F1:', {k:validation[k]['macro_f1'] for k in ('concatenation','gated')})
print('Validation improved:', improved, '\\nSeed summary:', selection)
''')
    md('''## 6. Held-out evaluation of the final selected model
Do not tune settings after this cell. Completed test results are reused on recovery. Test data never select epochs, seeds, thresholds or warning policy.
''')
    code('''import pandas as pd
assert (RUN / 'final_selection.json').is_file()
test_path = RUN / 'results/test/metrics.json'
if test_path.exists() and (test_path.parent / 'complete.json').exists():
    test = json.loads(test_path.read_text())
else:
    original_test = verify_saved_predictions(data['test'], heads, CLIP, 'test', 'cuda')
    test = evaluate_selected(gated, data['test'], original_test, threshold, RUN, 'test', 'cuda')
comparison = pd.DataFrame([dict(model=name, validation_macro_f1=validation[key]['macro_f1'],
    test_accuracy=test[key]['accuracy'], test_macro_f1=test[key]['macro_f1'], test_roc_auc=test[key]['roc_auc'])
    for name,key in [('DistilBERT + CLIP concatenation','concatenation'),('DistilBERT + CLIP gated','gated')]])
comparison.to_csv(RUN / 'comparison.csv', index=False)
display(comparison)
report = dict(validation=validation, test=test, validation_improved=improved,
    baseline_ablation=baseline_ablation, gated_ablation=gated_ablation,
    interpretation='Compare measured swap sensitivity, not just gate values. OOD interventions do not establish causal importance or robustness to misinformation.',
    deployment='Current application unchanged; CPU export verification required.')
save_json(report, RUN / 'evaluation_report.json')
from IPython.display import display, Image as DisplayImage
for path in [RUN/'ablations/concatenation_validate/ablation.png', RUN/'ablations/gated_validate/ablation.png', *sorted((RUN/'results/test').glob('*.png'))]:
    display(DisplayImage(filename=str(path)))
''')
    md('''## 7. Separate Windows CPU export and raw-input parity checks
Supply the **original** frozen encoder assets. Missing assets stop export. Check three predeclared validation examples against cached-feature output and require identical repeated CPU predictions. Cold-load and inference timings are saved. A ZIP is created only after these checks pass; the app remains unchanged.
''')
    code('''from gated_export import export_bundle, cpu_smoke
export_dir = export_bundle(RUN, BASELINE, CLIP, TEXT_ASSETS, CLIP_ASSETS, WORK)
predictor = cpu_smoke(export_dir, frames['validate'], data['validate'], gated, RUN / 'cpu_verification.json')
shutil.copy2(RUN / 'cpu_verification.json', export_dir / 'cpu_verification.json')
archive = shutil.make_archive(str(RUN / 'gated_windows_export'), 'zip', root_dir=export_dir)
print('Verified CPU bundle:', archive)
print('Download from Drive; keep separate from the current application.')
''')
    md('''## 8. Optional Fargo qualitative diagnostic — after selection and evaluation
Set FARGO_IMAGE above to the existing image, or upload **flamethrower.jpg** here. Never adjust settings in response to this result. User supplies the Fake label; this is not part of benchmark performance. Gate values are learned coefficients, not definitive feature importance.
''')
    code('''RUN_FARGO = False  # Set True to upload/run the diagnostic after completing cells above.
if RUN_FARGO:
    if not FARGO_IMAGE:
        uploaded = files.upload()
        assert len(uploaded) == 1, 'Upload the supplied flamethrower image only.'
        FARGO_IMAGE = str(WORK / next(iter(uploaded)))
        del uploaded
    assert not any(sha_file(FARGO_IMAGE) in set(frame.image_sha256) for frame in frames.values()), 'Diagnostic appears in data; stop.'
    from PIL import Image
    with Image.open(FARGO_IMAGE) as image:
        result = predictor.predict(FARGO_TITLE, image)
    save_json(dict(headline=FARGO_TITLE, label=0, label_source='User supplied', qualitative_only=True,
        image_sha256=sha_file(FARGO_IMAGE), measured=result,
        previously_reported_fake=dict(text=.004,resnet_image=.75,resnet_fusion=.163,clip_image=.907,clip_fusion=.06)), RUN / 'fargo_diagnostic.json')
    print(json.dumps(result, indent=2))
else:
    print('Fargo diagnostic skipped; benchmark results are unaffected.')
''')
    md('''## 9. What to retain and what to do next
Drive run folder contains the actual trained `gated_fusion.pt`, `selection.json`, configuration, epoch recovery files, evaluation report, comparison CSV, plots, intervention tables, threshold/warning policy and verified Windows ZIP.
No trained gated checkpoint is supplied with this notebook before execution. Missing caches must be restored from the original Drive experiment; do not substitute Windows checkpoints for feature shards.
Only after inspecting **validation** improvement and CPU parity, create a separate app copy to add an optional sixth model. Instantiate/cache `GatedPredictor` once; reuse it for predictions. Display warnings without overriding labels. Keep CLIP fusion as default unless promotion is justified. The existing application requires no rollback because this workflow never changes it.
''')
    notebook = dict(cells=cells, metadata=dict(accelerator='GPU', kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python',version='3.12')), nbformat=4, nbformat_minor=5)
    for i,cell in enumerate(cells):
        cell['id'] = f'gated-{i:02}'
    (HERE / 'colab_gated_fusion.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    return notebook


if __name__ == '__main__':
    build()
