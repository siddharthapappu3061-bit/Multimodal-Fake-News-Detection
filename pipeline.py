"""GPU-only bulk extraction/training; persistent, fingerprinted recovery files."""
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from models import ENCODER_CONFIG, Head


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fingerprint(frames, config):
    records = {s: df[["id", "clean_title", "2_way_label", "image_sha256"]].to_dict("records") for s, df in frames.items()}
    return hashlib.sha256(json.dumps({"data": records, "config": config}, sort_keys=True).encode()).hexdigest()[:20]


def gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("Select Colab Runtime > Change runtime type > GPU, then rerun. Bulk CPU work is disabled.")
    return torch.device("cuda")


def extract(frames, encoders, cache, batch_size=32):
    gpu()
    cache = Path(cache)
    result = {}
    for split, df in frames.items():
        # Separate modality chunks let a completed text batch survive an image failure.
        features = {}
        for modality, dim in (("text", 768), ("image", 2048)):
            chunks = []
            for start in range(0, len(df), batch_size):
                batch = df.iloc[start:start + batch_size]
                ids = batch.id.tolist()
                labels = torch.tensor(batch["2_way_label"].tolist(), dtype=torch.long)
                path = cache / split / modality / f"{start:06d}-{len(batch):04d}.pt"
                part = None
                if path.exists():
                    try:
                        saved = torch.load(path, map_location="cpu", weights_only=True)
                        if (saved["ids"] == ids and torch.equal(saved["labels"], labels)
                                and saved["features"].shape == (len(batch), dim)
                                and torch.isfinite(saved["features"]).all()):
                            part = saved
                    except (OSError, RuntimeError, EOFError, ValueError, KeyError, pickle.UnpicklingError):
                        print("Recomputing unreadable chunk:", path)
                if part is None:
                    if modality == "text":
                        embedding = encoders.encode_text(batch.clean_title.tolist())
                    else:
                        images = []
                        for filename in batch.resolved_path:
                            with Image.open(filename) as im:
                                images.append(im.convert("RGB"))
                        embedding = encoders.encode_images(images)
                    part = {"ids": ids, "labels": labels, "features": embedding.cpu()}
                    if not torch.isfinite(part["features"]).all():
                        raise ValueError("Non-finite embeddings")
                    atomic_save(part, path)
                chunks.append(part["features"])
            features[modality] = torch.cat(chunks)
            print(f"{split}/{modality}: {len(df)} cached features ready")
        result[split] = {**features, "labels": torch.tensor(df["2_way_label"].tolist(), dtype=torch.long), "ids": df.id.tolist()}
    return result


def train(kind, features, run, config, run_id):
    device = gpu()
    run = Path(run)
    last_path = run / "recovery" / f"{kind}.pt"
    best_path = run / "models" / f"{kind}.pt"
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = Head(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    start, best, stale, history, best_state = 0, float("inf"), 0, [], None
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if state["run_id"] != run_id:
            raise ValueError("Recovery fingerprint mismatch")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        for opt in optimizer.state.values():
            for key, value in opt.items():
                if isinstance(value, torch.Tensor):
                    opt[key] = value.to(device)
        start, best, stale, history = state["epoch"] + 1, state["best"], state["stale"], state["history"]
        best_state = state["best_state"]
        torch.set_rng_state(state["rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"{kind}: recovering after epoch {start}")
    data = {s: {k: v.to(device) for k, v in features[s].items() if k != "ids"} for s in ("train", "validate")}
    for epoch in range(start, config["epochs"]):
        if stale >= config["patience"]:
            break
        model.train()
        order = torch.randperm(len(data["train"]["labels"]), device=device)
        total = 0.0
        for idx in order.split(config["batch_size"]):
            d = data["train"]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(d["text"][idx], d["image"][idx]), d["labels"][idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.inference_mode():
            d = data["validate"]
            val = loss_fn(model(d["text"], d["image"]), d["labels"]).item()
        if not np.isfinite(val):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "val_loss": val})
        if val < best:
            best, stale = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        # One recovery file contains both last and best states, avoiding split-file recovery races.
        atomic_save({"run_id": run_id, "epoch": epoch, "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "best_state": best_state, "best": best,
                     "stale": stale, "history": history, "rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all()}, last_path)
        print(f"{kind} epoch {epoch + 1}: train={history[-1]['train_loss']:.4f} val={val:.4f}")
    if best_state is None:
        raise ValueError("No trained checkpoint available; epochs must be positive")
    atomic_save({"state_dict": best_state, "kind": kind, "run_id": run_id}, best_path)
    write_json(history, run / "results" / f"{kind}_history.json")
    return history


def evaluate(features, run, run_id):
    device = gpu()
    run = Path(run)
    test = features["test"]
    assert len(test["ids"]) == 500, "Use the unchanged 500-example test split"
    y = test["labels"].numpy()
    metrics, predictions = {}, {"id": test["ids"], "label": y.tolist()}
    for kind in ("text", "image", "fusion"):
        checkpoint = torch.load(run / "models" / f"{kind}.pt", map_location="cpu", weights_only=True)
        assert checkpoint["run_id"] == run_id
        model = Head(kind).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.inference_mode():
            probabilities = model(test["text"].to(device), test["image"].to(device)).softmax(1).cpu().numpy()
        pred = probabilities.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=0, zero_division=0)
        metrics[kind] = {"accuracy": float(accuracy_score(y, pred)), "precision": float(p), "recall": float(r),
                         "f1": float(f), "roc_auc": float(roc_auc_score(y == 0, probabilities[:, 0])),
                         "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
        predictions[f"{kind}_prediction"] = pred.tolist()
        predictions[f"{kind}_fake_probability"] = probabilities[:, 0].tolist()
    result = {"run_id": run_id, "test_count": 500, "positive_class": "Fake (0)", "models": metrics,
              "fusion_outperforms_both": {metric: all(metrics["fusion"][metric] > metrics[k][metric] for k in ("text", "image"))
                                          for metric in ("accuracy", "f1")}}
    write_json(result, run / "results" / "metrics.json")
    import pandas as pd
    pd.DataFrame(predictions).to_csv(run / "results" / "predictions.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")  # Export figures without a desktop/Tk dependency.
    import matplotlib.pyplot as plt
    pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot.bar(ylim=(0, 1), figsize=(10, 5))
    plt.tight_layout()
    plt.savefig(run / "results" / "comparison.png")
    plt.close()
    for kind in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        matrix = np.array(metrics[kind]["confusion_matrix"])
        axes[0].imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            axes[0].text(j, i, str(value), ha="center")
        axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"], xlabel="Predicted", ylabel="Actual")
        history = pd.read_json(run / "results" / f"{kind}_history.json")
        axes[1].plot(history.epoch, history.train_loss, label="Train")
        axes[1].plot(history.epoch, history.val_loss, label="Validation")
        axes[1].legend()
        axes[1].set(xlabel="Epoch", ylabel="Cross-entropy")
        fig.suptitle(kind)
        fig.tight_layout()
        fig.savefig(run / "results" / f"{kind}.png")
        plt.close(fig)
    return result
