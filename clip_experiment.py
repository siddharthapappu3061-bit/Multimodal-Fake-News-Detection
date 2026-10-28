"""Isolated GPU experiment. Baseline modules, caches, and checkpoints are read-only."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score, confusion_matrix

from models import Head, ENCODER_CONFIG
from pipeline import atomic_save, write_json, fingerprint, gpu
from clip_models import ClipEncoder, ClipHead, FEATURE_SPEC, CLIP_ID

EXPERIMENT_VERSION = 1
KINDS = ("clip_image", "clip_fusion")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def sha_file(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def guard_output(output, *protected):
    output = Path(output).resolve()
    for path in protected:
        path = Path(path).resolve()
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError(f"Experiment output must be separate from protected path: {path}")
    return output


def read_baseline_config(run, frames):
    run = Path(run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if config["encoders"] != ENCODER_CONFIG or config["labels"] != {"0": "Fake", "1": "Real"}:
        raise ValueError("Baseline encoder or label contract differs from this experiment")
    expected = fingerprint(frames, config["encoders"])
    if config["feature_id"] != expected:
        raise ValueError("Baseline dataset fingerprint mismatch; do not mix different processed datasets")
    provenance = {"config": config, "checkpoint_sha256": {}}
    for kind in ("text", "image", "fusion"):
        path = run / "models" / f"{kind}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["run_id"] != config["run_id"] or saved["kind"] != kind:
            raise ValueError("Baseline checkpoints do not belong to the selected run")
        provenance["checkpoint_sha256"][kind] = sha_file(path)
    return config, provenance


def read_cached_modality(folder, frame, dimension):
    """Join by ID and validate labels; never trust chunk order or row positions."""
    rows = {}
    paths = sorted(Path(folder).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No existing features in {folder}. Restore the original Drive feature cache; this experiment does not regenerate it.")
    for path in paths:
        part = torch.load(path, map_location="cpu", weights_only=True)
        ids, labels, features = part["ids"], part["labels"], part["features"]
        if (features.shape != (len(ids), dimension) or labels.shape != (len(ids),)
                or labels.dtype != torch.long or not torch.isfinite(features).all()):
            raise ValueError(f"Invalid cached feature dimensions/values: {path}")
        for index, sample_id in enumerate(ids):
            label, embedding = int(labels[index]), features[index].float()
            if sample_id in rows:
                old_label, old_embedding = rows[sample_id]
                if label != old_label or not torch.allclose(embedding, old_embedding, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Conflicting duplicate cache ID: {sample_id}")
            rows[sample_id] = (label, embedding)
    if set(rows) != set(frame.id) or len(frame.id) != len(set(frame.id)):
        raise ValueError(f"Missing/extra IDs in {folder}; cannot preserve split alignment")
    expected = frame["2_way_label"].astype(int).tolist()
    if [rows[sample_id][0] for sample_id in frame.id] != expected:
        raise ValueError(f"Cached labels disagree with CSV in {folder}")
    return torch.stack([rows[sample_id][1] for sample_id in frame.id])


def load_baseline_features(cache_root, feature_id, frames):
    result = {}
    for split, frame in frames.items():
        folder = Path(cache_root) / feature_id / split
        result[split] = {
            "ids": frame.id.tolist(), "labels": torch.tensor(frame["2_way_label"].tolist(), dtype=torch.long),
            "text": read_cached_modality(folder / "text", frame, 768),
            "image": read_cached_modality(folder / "image", frame, 2048),
        }
    return result


def prepare_clip(output, revision="main"):
    """Resolve once to an immutable HF revision; future recovery is offline."""
    gpu()
    output = Path(output)
    request = {"model": CLIP_ID, "requested_revision": revision}
    root = output / "encoders" / digest(request)
    marker = root / "READY.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["request"] != request or manifest["feature_spec"] != FEATURE_SPEC:
            raise ValueError("CLIP asset manifest mismatch")
        encoder = ClipEncoder(root, "cuda", local=True)
    else:
        from huggingface_hub import model_info
        commit = model_info(CLIP_ID, revision=revision).sha
        encoder = ClipEncoder(CLIP_ID, "cuda", revision=commit, local=False)
        root.mkdir(parents=True, exist_ok=True)
        encoder.save(root)
        manifest = {"request": request, "resolved_revision": commit, "feature_spec": FEATURE_SPEC,
                    "sha256": {p.name: sha_file(p) for p in root.iterdir()
                               if p.is_file() and not p.name.endswith(".tmp")}}
        write_json(manifest, marker)
    for name, expected in manifest["sha256"].items():
        if sha_file(root / name) != expected:
            raise ValueError(f"CLIP asset changed: {name}; restore it or use a new experiment output directory")
    return encoder, root, manifest


def extract_clip(frames, encoder, folder, feature_id, batch_size=32):
    gpu()
    if batch_size < 1:
        raise ValueError("Feature batch size must be positive")
    folder = Path(folder) / feature_id
    result = {}
    for split, frame in frames.items():
        # Fixed shards are independent of GPU batch size: lowering the batch size retains completed work.
        chunks = []
        for start in range(0, len(frame), 128):
            batch = frame.iloc[start:start + 128]
            labels = torch.tensor(batch["2_way_label"].tolist(), dtype=torch.long)
            ids = batch.id.tolist()
            path = folder / split / f"{start:06d}.pt"
            saved = None
            if path.exists():
                try:
                    saved = torch.load(path, map_location="cpu", weights_only=True)
                    if (saved["feature_id"] != feature_id or saved["ids"] != ids or not torch.equal(saved["labels"], labels)
                            or saved["features"].shape != (len(ids), encoder.dimension)
                            or not torch.isfinite(saved["features"]).all()):
                        saved = None
                except (OSError, RuntimeError, EOFError, ValueError, KeyError, pickle.UnpicklingError):
                    saved = None
            if saved is None:
                embeddings = []
                for offset in range(0, len(batch), batch_size):
                    images = []
                    for filename in batch.iloc[offset:offset + batch_size].resolved_path:
                        with Image.open(filename) as image:
                            images.append(image.copy())
                    embeddings.append(encoder.encode(images).cpu())
                feature = torch.cat(embeddings)
                if feature.shape != (len(ids), encoder.dimension) or not torch.isfinite(feature).all():
                    raise ValueError("Invalid CLIP embeddings")
                saved = {"feature_id": feature_id, "ids": ids, "labels": labels, "features": feature}
                atomic_save(saved, path)
            chunks.append(saved["features"])
        result[split] = torch.cat(chunks)
        print(f"CLIP {split}: {len(frame)} features ready")
    return result


def paired_features(baseline, visual):
    return {split: {**data, "image": visual[split]} for split, data in baseline.items()}


def cache_hashes(features):
    return {split: {modality: hashlib.sha256(data[modality].contiguous().numpy().tobytes()).hexdigest()
                    for modality in ("text", "image")} for split, data in features.items()}


def binary_metrics(labels, probabilities):
    y = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(y), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid binary probabilities")
    pred = probabilities.argmax(1)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=0, zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y == 0, probabilities[:, 0])) if len(set(y)) == 2 else None,
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


@torch.inference_mode()
def probabilities(model, data, device):
    model.to(device).eval()
    output = []
    for start in range(0, len(data["labels"]), 128):
        logits = model(data["text"][start:start + 128].to(device), data["image"][start:start + 128].to(device))
        output.append(logits.softmax(1).cpu())
    return torch.cat(output).numpy()


def train_trial(kind, data, root, config, run_id):
    """Validation macro-F1 selects epochs; validation loss breaks exact ties."""
    device = gpu()
    spec = dict(kind=kind, text_dim=data["train"]["text"].shape[1], image_dim=data["train"]["image"].shape[1], dropout=config["dropout"])
    trial_id = digest({"run_id": run_id, "spec": spec, "training": config})
    root = Path(root) / "trials" / kind / trial_id
    last = root / "recovery.pt"
    root.mkdir(parents=True, exist_ok=True)
    if config["epochs"] < 1 or config["patience"] < 1 or config["batch_size"] < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ClipHead(**spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    start, stale, best_score, best_loss, best_state, history = 0, 0, -1., float("inf"), None, []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if state["trial_id"] != trial_id or state["run_id"] != run_id:
            raise ValueError("Trial recovery mismatch")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        # AdamW maps moment buffers to the parameter device during load_state_dict.
        start, stale = state["epoch"] + 1, state["stale"]
        best_score, best_loss, best_state, history = state["best_score"], state["best_loss"], state["best_state"], state["history"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    # Test tensors are never accessed by training or trial selection.
    train = {k: data["train"][k].to(device) for k in ("text", "image", "labels")}
    val = {k: data["validate"][k].to(device) for k in ("text", "image", "labels")}
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(start, config["epochs"]):
        if stale >= config["patience"]:
            break
        model.train()
        order = torch.randperm(len(train["labels"]), device=device)
        total = 0.
        for ids in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train["text"][ids], train["image"][ids]), train["labels"][ids])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
        model.eval()
        with torch.inference_mode():
            logits = model(val["text"], val["image"])
            val_loss = loss_fn(logits, val["labels"]).item()
            metric = binary_metrics(val["labels"].cpu().numpy(), logits.softmax(1).cpu().numpy())
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss")
        score = metric["macro_f1"]
        if (score, -val_loss) > (best_score, -best_loss):
            best_score, best_loss, stale = score, val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "validation_loss": val_loss, "validation_macro_f1": score})
        atomic_save({"run_id": run_id, "trial_id": trial_id, "epoch": epoch, "stale": stale,
                     "best_score": best_score, "best_loss": best_loss, "best_state": best_state, "history": history,
                     "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, last)
        print(f"{kind} {trial_id} epoch {epoch + 1}: validation macro-F1={score:.4f}")
    if best_state is None:
        raise ValueError("No completed training epoch")
    checkpoint = {"run_id": run_id, "trial_id": trial_id, "spec": spec, "state_dict": best_state,
                  "training": config, "validation_macro_f1": best_score, "validation_loss": best_loss}
    atomic_save(checkpoint, root / "best.pt")
    write_json(history, root / "history.json")
    return root / "best.pt"


def select_trials(data, run, configs, run_id):
    run = Path(run)
    selected = {}
    for kind in KINDS:
        candidates = [train_trial(kind, data, run, cfg, run_id) for cfg in configs]
        records = [torch.load(path, map_location="cpu", weights_only=True) for path in candidates]
        index = max(range(len(records)), key=lambda i: (records[i]["validation_macro_f1"], -records[i]["validation_loss"]))
        saved = records[index]
        atomic_save(saved, run / "models" / f"{kind}.pt")
        selected[kind] = {k: saved[k] for k in ("trial_id", "spec", "training", "validation_macro_f1", "validation_loss")}
        selected[kind]["all_trials"] = [{k: r[k] for k in ("trial_id", "training", "validation_macro_f1", "validation_loss")} for r in records]
    write_json({"selection_metric": "validation macro-F1; ties: lower validation loss, then grid order", "selected": selected}, run / "selection.json")
    return selected


def compare_models(baseline, clip_data, baseline_run, run, run_id, splits=("validate",)):
    device = gpu()
    run, baseline_run = Path(run), Path(baseline_run)
    (run / "results").mkdir(parents=True, exist_ok=True)
    results = {}
    base_config = json.loads((baseline_run / "config.json").read_text())
    for split in splits:
        if split not in ("validate", "test") or len(baseline[split]["ids"]) != 500:
            raise ValueError("Comparison requires the unchanged 500-example validation/test split")
        if baseline[split]["ids"] != clip_data[split]["ids"] or not torch.equal(baseline[split]["labels"], clip_data[split]["labels"]):
            raise ValueError("Comparison split pairing mismatch")
        rows, metrics = {}, {}
        for name, old_kind in (("distilbert", "text"), ("resnet50", "image"), ("distilbert_resnet50", "fusion"), ("clip_image", None), ("clip_fusion", None)):
            if old_kind:
                saved = torch.load(baseline_run / "models" / f"{old_kind}.pt", map_location="cpu", weights_only=True)
                if saved["run_id"] != base_config["run_id"] or saved["kind"] != old_kind:
                    raise ValueError("Invalid baseline checkpoint")
                model, data = Head(old_kind), baseline[split]
            else:
                saved = torch.load(run / "models" / f"{name}.pt", map_location="cpu", weights_only=True)
                if saved["run_id"] != run_id or saved["spec"]["kind"] != name:
                    raise ValueError("Invalid CLIP checkpoint")
                model, data = ClipHead(**saved["spec"]), clip_data[split]
            model.load_state_dict(saved["state_dict"])
            probs = probabilities(model, data, device)
            metrics[name] = binary_metrics(data["labels"].numpy(), probs)
            rows[f"{name}_prediction"] = probs.argmax(1)
            rows[f"{name}_fake_probability"] = probs[:, 0]
            del model
        pd.DataFrame({"id": baseline[split]["ids"], "label": baseline[split]["labels"].tolist(), **rows}).to_csv(run / "results" / f"{split}_predictions.csv", index=False)
        result = {"run_id": run_id, "split": split, "count": 500, "positive_class": "Fake (0)",
                  "confusion_order": ["Fake", "Real"], "models": metrics}
        write_json(result, run / "results" / f"{split}_metrics.json")
        pd.DataFrame(metrics).T.drop(columns="confusion_matrix").to_csv(run / "results" / f"{split}_comparison.csv")
        results[split] = result
    return results


def promotion_gate(validation, minimum_gain=0.0):
    if validation["split"] != "validate" or minimum_gain < 0:
        raise ValueError("Promotion uses validation only and nonnegative minimum gain")
    metrics = validation["models"]
    comparisons = {"clip_image": "resnet50", "clip_fusion": "distilbert_resnet50"}
    return {"criterion": "strict validation macro-F1 improvement", "minimum_gain": minimum_gain,
            "candidates": {kind: {"baseline": old, "delta_macro_f1": metrics[kind]["macro_f1"] - metrics[old]["macro_f1"],
                                   "eligible": metrics[kind]["macro_f1"] - metrics[old]["macro_f1"] > minimum_gain}
                           for kind, old in comparisons.items()},
            "note": "Eligibility is an observed validation improvement, not statistical significance. Test metrics do not change this decision."}


def plot_comparison(run, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run)
    result = json.loads((run / "results" / f"{split}_metrics.json").read_text())
    frame = pd.DataFrame(result["models"]).T.drop(columns="confusion_matrix").astype(float)
    frame.plot.bar(ylim=(0, 1), figsize=(12, 5))
    plt.tight_layout()
    plt.savefig(run / "results" / f"{split}_comparison.png")
    plt.close()
    fig, axes = plt.subplots(1, len(result["models"]), figsize=(17, 3.5))
    for ax, (name, metric) in zip(axes, result["models"].items()):
        matrix = np.array(metric["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(j, i, str(count), ha="center")
        ax.set(title=name, xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1], xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    fig.tight_layout()
    fig.savefig(run / "results" / f"{split}_confusion_matrices.png")
    plt.close(fig)


def export_bundle(run, clip_assets, text_assets, destination, config, source_dir):
    """Separate clip_artifacts package; no production files or baseline checkpoint copies."""
    run, destination = Path(run), Path(destination)
    for name in ("validate_metrics.json", "test_metrics.json", "promotion.json"):
        if not (run / "results" / name).exists():
            raise FileNotFoundError(f"Complete evaluation before export: {name}")
    bundle = destination / "clip_artifacts"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in ("models", "results"):
        shutil.copytree(run / name, bundle / name, dirs_exist_ok=True)
    shutil.copytree(clip_assets, bundle / "encoders/clip", dirs_exist_ok=True)
    shutil.copytree(text_assets, bundle / "encoders/distilbert", dirs_exist_ok=True)
    write_json(config, bundle / "config.json")
    shutil.copy2(run / "selection.json", bundle / "selection.json")
    for name in ("clip_models.py", "clip_inference.py", "clip_requirements.txt", "CLIP_EXPERIMENT.md"):
        shutil.copy2(Path(source_dir) / name, destination / name)
    return bundle
