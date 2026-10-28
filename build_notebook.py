"""Regenerate the self-contained Colab notebook after changing shared source files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)})


md("""# Multimodal Fake News Detection — Colab GPU training
Run cells in order. Select **Runtime → Change runtime type → T4 GPU** first.
Only pretrained encoder weights and Python dependencies are downloaded; **Fakeddit is never downloaded**.
Use your existing 3,000/500/500 processed pairs. Backbones remain frozen; only classification heads train.
Drive stores cached features, epoch recovery, models, and results. Keep the same settings to resume after disconnecting.
Upload this notebook to Colab; it embeds all required Python source, so no repository upload is needed.
""")
md("## 1. Install dependencies (keep Colab's matched CUDA torch/torchvision)")
code('''%pip -q install "transformers==4.57.1" "pandas>=2.2,<3" "scikit-learn>=1.5,<2" "matplotlib>=3.8,<4" "Pillow>=10,<13" safetensors
import torch, torchvision
assert torch.cuda.is_available(), "Enable a Colab GPU runtime before continuing."
print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__, "torchvision:", torchvision.__version__)
''')
md("## 2. Mount Drive and write the shared training/inference code")
# Keep the original training export standalone; the integrated app reads two separate bundles.
embedded = {name: (ROOT / ("baseline_app.py" if name == "app.py" else name)).read_text(encoding="utf-8") for name in ("dataset.py", "models.py", "pipeline.py", "inference.py", "app.py", "requirements.txt")}
code('''from google.colab import drive, files
from pathlib import Path
import os, json, shutil, sys
drive.mount('/content/drive')
WORK = Path('/content/fakeddit_code')
WORK.mkdir(exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
SOURCE_FILES = ''' + repr(embedded) + '''
for name, source in SOURCE_FILES.items():
    (WORK / name).write_text(source, encoding="utf-8")
''')
md("""## 3. Configure data source and persistent output
`drive`: point DATA_DIR at the folder containing train_processed.csv and images/.
`zip`: upload fakeddit_processed.zip when prompted, or set ZIP_ON_DRIVE to an existing ZIP path.
Use package_data.py on Windows to create the ZIP without downloading anything.
Only settings in this cell need editing. Existing CSVs are never rewritten.
""")
code('''DATA_SOURCE = "drive"  # "drive" or "zip"
DATA_DIR = Path('/content/drive/MyDrive/Multimodal_Fake_News_Detection/data')
ZIP_ON_DRIVE = ""  # e.g. /content/drive/MyDrive/fakeddit_processed.zip
OUTPUT = Path('/content/drive/MyDrive/Fakeddit_Demo')
TRAIN_CONFIG = dict(seed=42, epochs=25, batch_size=64, lr=0.001, weight_decay=0.0001, patience=5)
FEATURE_BATCH_SIZE = 32  # Lower to 16 or 8 if GPU runs out of memory.
OUTPUT.mkdir(parents=True, exist_ok=True)
from dataset import extract_zip, validate
if DATA_SOURCE == "zip":
    if ZIP_ON_DRIVE:
        archive = Path(ZIP_ON_DRIVE)
    else:
        uploaded = files.upload()
        assert len(uploaded) == 1, "Upload exactly one processed dataset ZIP."
        archive = WORK / next(iter(uploaded))
        del uploaded
    DATA_DIR = extract_zip(archive, Path('/content/fakeddit_data'))
elif DATA_SOURCE != "drive":
    raise ValueError('DATA_SOURCE must be drive or zip')
assert (DATA_DIR / 'train_processed.csv').is_file(), f'Check DATA_DIR: {DATA_DIR}'
''')
md("""## 4. Validate every pair and preserve the existing splits
Checks counts, labels, missing fields, duplicate IDs, overlap, image existence and decoding.
Stops on errors without dropping rows. Windows backslashes and absolute paths relocate via images/.
Binary labels stay 0=Fake, 1=Real; verify against the supplied 6_way_label true category (0).
References: [official Fakeddit repository](https://github.com/entitize/Fakeddit) and [paper](https://aclanthology.org/2020.lrec-1.755/).
""")
code('''from pipeline import fingerprint, write_json
from models import ENCODER_CONFIG, HEAD_CONFIG
frames, report = validate(DATA_DIR, OUTPUT / 'dataset_validation.json')
FEATURE_ID = fingerprint(frames, ENCODER_CONFIG)
RUN_ID = fingerprint(frames, {"encoders": ENCODER_CONFIG, "heads": HEAD_CONFIG, "training": TRAIN_CONFIG})
RUN = OUTPUT / 'runs' / RUN_ID
RUN.mkdir(parents=True, exist_ok=True)
write_json(report, RUN / 'results' / 'dataset_validation.json')
config = {"run_id": RUN_ID, "feature_id": FEATURE_ID, "encoders": ENCODER_CONFIG, "heads": HEAD_CONFIG,
          "training": TRAIN_CONFIG, "labels": {"0": "Fake", "1": "Real"}}
write_json(config, RUN / 'config.json')
print(json.dumps(report, indent=2))
print('Persistent run:', RUN)
''')
md("## 5. Load frozen pretrained encoders and extract/reuse GPU features\nEach completed modality batch is persisted to Drive. Restart cells 1–5 to resume. Changing training settings reuses the same feature cache. Keep FEATURE_BATCH_SIZE unchanged when recovering feature extraction.")
code('''from models import Encoders
from pipeline import extract
ASSETS = OUTPUT / 'encoders_v1'
ready = ASSETS / 'READY.json'
encoders = Encoders(ASSETS, device='cuda', download=not ready.exists())
if ready.exists():
    assert json.loads(ready.read_text()) == ENCODER_CONFIG
else:
    write_json(ENCODER_CONFIG, ready)
assert all(not p.requires_grad for m in (encoders.text, encoders.image) for p in m.parameters())
features = extract(frames, encoders, OUTPUT / 'features' / FEATURE_ID, FEATURE_BATCH_SIZE)
del encoders
import gc
gc.collect()
torch.cuda.empty_cache()
''')
md("## 6. Smoke-test shapes and serialization before training")
code('''from models import Head, masked_mean
import tempfile
assert torch.equal(masked_mean(torch.tensor([[[1., 2.], [99., 99.]]]), torch.tensor([[1, 0]])), torch.tensor([[1., 2.]]))
for split, data in features.items():
    assert data['text'].shape == (len(frames[split]), 768)
    assert data['image'].shape == (len(frames[split]), 2048)
    assert data['ids'] == frames[split].id.tolist()
for kind in ('text', 'image', 'fusion'):
    head = Head(kind).eval()
    t, i = features['train']['text'][:2], features['train']['image'][:2]
    assert head(t, i).shape == (2, 2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'head.pt'
        torch.save(head.state_dict(), path)
        restored = Head(kind).eval()
        restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        torch.testing.assert_close(head(t, i), restored(t, i))
print('Shape, masked pooling and CPU checkpoint round-trip checks passed.')
''')
md("## 7. Train three independent heads on GPU\nOnly train fits parameters; validate selects the lowest-loss checkpoint. Recovery includes optimizer, RNG, early stopping and the best weights. An interrupted epoch is replayed; completed epochs are retained. Completed runs are skipped automatically.")
code('''from pipeline import train
for kind in ('text', 'image', 'fusion'):
    train(kind, features, RUN, TRAIN_CONFIG, RUN_ID)
''')
md("## 8. Evaluate all models on the same unchanged 500 test examples\nFake (0) is the positive class for precision/recall/F1/AUC. No test-based model selection. Results below are measured, never placeholders.")
code('''from pipeline import evaluate
metrics = evaluate(features, RUN, RUN_ID)
print(json.dumps(metrics, indent=2))
from IPython.display import display, Image as DisplayImage
display(DisplayImage(filename=str(RUN / 'results' / 'comparison.png')))
''')
md("## 9. Export encoders, heads, configs and results; verify offline CPU inference\nThe Windows bundle contains both pretrained backbones so the demo needs no model downloads. A single-example CPU check compares exported inference with the GPU cached-feature prediction.")
code('''import importlib.metadata
import zipfile
from PIL import Image
from inference import Predictor
BUNDLE = Path('/content/windows_demo/artifacts')
BUNDLE.mkdir(parents=True, exist_ok=True)
for folder in ('models', 'results'):
    shutil.copytree(RUN / folder, BUNDLE / folder, dirs_exist_ok=True)
shutil.copy2(RUN / 'config.json', BUNDLE / 'config.json')
shutil.copytree(ASSETS, BUNDLE / 'encoders', dirs_exist_ok=True)
versions = {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'transformers', 'numpy', 'pandas', 'Pillow', 'scikit-learn')}
write_json(versions, BUNDLE / 'environment.json')
torch.set_num_threads(2)
predictor = Predictor(BUNDLE)
row = frames['test'].iloc[0]
with Image.open(row.resolved_path) as im:
    prediction = predictor.predict(row.clean_title, im)
for kind, head in predictor.heads.items():
    with torch.inference_mode():
        expected = head(features['test']['text'][:1], features['test']['image'][:1]).softmax(1)[0]
    torch.testing.assert_close(torch.tensor(prediction[kind]), expected, atol=2e-3, rtol=2e-3)
write_json({'passed': True, 'sample_id': row.id, 'probabilities': prediction}, BUNDLE / 'results' / 'cpu_smoke.json')
for name in ('app.py', 'models.py', 'dataset.py', 'inference.py', 'requirements.txt'):
    shutil.copy2(WORK / name, BUNDLE.parent / name)
export_path = OUTPUT / 'windows_demo.zip'
temporary = export_path.with_suffix('.zip.tmp')
with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in BUNDLE.parent.rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(BUNDLE.parent).as_posix())
os.replace(temporary, export_path)
shutil.copy2(BUNDLE / 'results' / 'cpu_smoke.json', RUN / 'results' / 'cpu_smoke.json')
print('CPU export check passed. Download from Google Drive:', export_path)
print('Extract ZIP into your Windows project; then run: python -m streamlit run app.py')
''')
md("""## 10. Optional browser download
The ZIP is already saved in Drive. For large bundles, downloading through the Drive website is more reliable.
After a Colab disconnect rerun the cells in order with identical settings. Never delete Drive recovery/cache folders to resume.
No full training or feature extraction is required on Windows.
""")
code("# Optional:\n# files.download(str(export_path))\n")
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "colab": {"name": "colab_training.ipynb"}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
for index, cell in enumerate(cells):
    cell['id'] = f'cell-{index:02d}'
(ROOT / 'colab_training.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
print('Wrote colab_training.ipynb')
