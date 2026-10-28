# Multimodal Fake News Detection

A PyTorch and Streamlit research prototype comparing six text, image, and multimodal classifiers on Fakeddit. The application runs shared frozen encoders on CPU and shows per-model predictions, class probabilities, modality disagreement, and learned fusion gates.

> This is a dataset-based classification prototype, not an independent fact-checking service. Model probabilities are uncalibrated scores; they do not establish whether a news claim is true.

## Models

| Classifier | Features | Combination |
|---|---|---|
| DistilBERT | 768-dimensional text | Text classification head |
| ResNet50 | 2,048-dimensional image | Image classification head |
| DistilBERT + ResNet50 | Text and ResNet features | Project and concatenate |
| CLIP Vision | 512-dimensional normalized image | Image classification head |
| DistilBERT + CLIP Vision | Text and CLIP features | Project and concatenate |
| DistilBERT + CLIP gated fusion | Text and CLIP features | Learned modality gates and weighted concatenation |

The gated model projects each modality to 256 dimensions using Linear, LayerNorm, ReLU, and dropout. A small network produces two softmax gates. The weighted branches are concatenated and passed through a 512-to-128-to-2 classifier. Gates scale feature representations; they are not percentages of factual evidence. Plain fusion and gated fusion also differ in branch normalization and activation, so their comparison does not isolate the gate alone.

## Reported experiment

The existing experiment uses 3,000 training, 500 validation, and 500 test pairs. These are a processed subset, not the full Fakeddit dataset. Labels are `0 = Fake`, `1 = Real`.

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| DistilBERT + CLIP concatenation | 83.67% | 83.4% | 82.29% | 0.92499 |
| DistilBERT + CLIP gated fusion | 84.81% | 86.4% | 85.86% | 0.92637 |

These values are recorded in the existing local integration report; publication preparation did not retrain or rerun the models. Selected gated seed: 43. Checkpoints and per-sample outputs are not distributed here, so cloning this repository alone does not independently reproduce these numbers. Model selection uses validation data; test results must not guide threshold or hyperparameter tuning.

## Repository guide

| Path | Purpose |
|---|---|
| `app.py` / `six_model_app.py` | Current six-model Streamlit entrypoint and UI |
| `models.py`, `clip_models.py` | Frozen encoder interfaces and baseline classification heads |
| `gated_integration.py`, `gated_runtime.py` | Verified gated bundle loading and cached inference |
| `integrated_app.py`, `baseline_app.py` | Earlier application versions |
| `colab_training.ipynb` | Baseline feature extraction, training, and export |
| `colab_clip_experiment.ipynb` | CLIP experiment workflow |
| `experiments/gated_fusion/` | Gated model, cached-feature training, analysis, export, and notebook |
| `experiments/reliability*/`, `reliability.py` | Optional abstention analysis and display logic |
| `external_evaluation/` | Reusable external-evaluation tools and an empty sample template |
| `tests/` | Unit and artifact-dependent integration tests |

## Installation

Python 3.12 is recommended for the existing Windows workflow. From this repository's root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For Colab training, follow the notebook dependency cells and use a GPU runtime.

## Required trained artifacts

**The application needs trained bundles. They are excluded from Git, and the app is not runnable from a fresh clone until they are supplied.** It does not generate placeholder predictions or automatically download missing trained weights.

Supply complete, mutually compatible baseline, CLIP, and gated exports, including encoder files, classifier weights, configurations, checksum manifests, and evaluation reports. Set their locations with:

```powershell
$env:FAKEDDIT_BASELINE_BUNDLE = 'C:\path\to\baseline\artifacts'
$env:FAKEDDIT_CLIP_BUNDLE = 'C:\path\to\clip_artifacts'
$env:FAKEDDIT_GATED_BUNDLE = 'C:\path\to\gated_artifacts'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The loader checks artifact identities, label conventions, shared encoder weights, and prediction alignment. Gated default promotion additionally requires a matching successful `windows_integration_verification.json` beside the gated artifact directory. Without promotion evidence, the app defaults to CLIP concatenation when the remaining artifact checks pass.

On the original working machine, the existing launch command remains:

```powershell
.\CLIP-vision\.venv-clip\Scripts\python.exe -m streamlit run app.py
```

## Training and reproduction

1. Obtain Fakeddit from its [official repository](https://github.com/entitize/Fakeddit), review its usage conditions, and prepare paired text/image train, validation, and test CSVs expected by `dataset.py`. The repository does not redistribute the dataset.
2. Inspect notebook configuration cells and replace example/local paths. `config.yaml` documents baseline defaults; notebook cells define active run settings.
3. Run `colab_training.ipynb` to produce baseline features, heads, and an exported bundle.
4. Run `colab_clip_experiment.ipynb` for CLIP image and concatenation heads. Preserve the resulting feature caches and run metadata.
5. Run `experiments/gated_fusion/colab_gated_fusion.ipynb` with the matching baseline and CLIP runs, original processed data, verified feature shards, and encoder assets. This workflow expects the original 3,000/500/500 split contract; scaling data requires deliberate pipeline changes. Exported inference weights alone do not contain the training feature caches.
6. Export and verify the trained bundles before launching the application. Keep held-out test data outside model and threshold selection.

Notebook outputs and execution counts are cleared in this public source copy. Historical run identifiers and example paths in source may need configuration for a new environment.

## Reliability display

All six classifiers apply softmax to two logits. The UI displays the larger class score, not an accuracy estimate. The disagreement score is `abs(P_fake_text - P_fake_CLIP_image)`.

Selective classification can display `Requires Verification` instead of a binary decision when a validated acceptance policy rejects an input. The current saved 95% confidence policy failed the minimum 70% validation-coverage requirement and is unavailable. Missing or incompatible policies also disable the option; informational disagreement warnings remain available. Gate weights and disagreement are exploratory signals, not guarantees of correctness.

## Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Some tests require the original trained artifacts and exact experiment data, which are not in this source repository. A fresh clone is therefore not expected to pass the entire integration suite without those assets. Gated mechanics tests can be run with:

```powershell
$env:PYTHONPATH = "$PWD\experiments\gated_fusion;$PWD"
.\.venv\Scripts\python.exe -m unittest discover -s experiments\gated_fusion -p test_gated.py -v
```

Synthetic tests validate implementation mechanics; they are not model performance evidence.

## Limitations and excluded pilot

The main evaluation scope is held-out Fakeddit data. Generalization to news from other sources has not been established. A preliminary 40-sample external pilot was withdrawn from the main evaluation after review found verdict overlays or revealing interface elements in 19 original images. Its 57.5% gated result is not presented as a reliable external benchmark. Original pilot files remain archived locally and are excluded from this repository.

Dataset-specific patterns, label noise, domain shift, correlated examples, and confidently incorrect predictions remain limitations. The system does not retrieve supporting evidence or verify real-world events.

## Data, weights, and licensing

Datasets, third-party images, trained weights, caches, local environments, and archived pilot material are not included. Obtain third-party assets from their respective sources and follow their usage terms. No project-wide open-source license has been selected yet; public visibility alone does not grant an open-source license.

Reference: Nakamura, Levy, and Wang (2020), [Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://aclanthology.org/2020.lrec-1.755/).
