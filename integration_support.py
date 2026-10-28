"""Paths and verified report contracts for the five-model Windows application."""
import hashlib
import json
import math
import os
from pathlib import Path

MODEL_NAMES = {
    "text": "DistilBERT text-only",
    "image": "ResNet50 image-only",
    "fusion": "DistilBERT + ResNet50",
    "clip_image": "CLIP Vision image-only",
    "clip_fusion": "DistilBERT + CLIP Vision",
}
DEFAULT_MODEL = "clip_fusion"
REPORT_KEYS = {"text": "distilbert", "image": "resnet50", "fusion": "distilbert_resnet50",
               "clip_image": "clip_image", "clip_fusion": "clip_fusion"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bundle_paths(project):
    project = Path(project).resolve()
    baseline = project / "artifacts"
    if not baseline.exists():
        baseline = project / "windows_demo/artifacts"
    clip = project / "clip_artifacts"
    if not clip.exists():
        clip = project / "CLIP-vision/clip_artifacts"
    return (Path(os.environ.get("FAKEDDIT_BASELINE_BUNDLE", baseline)).resolve(),
            Path(os.environ.get("FAKEDDIT_CLIP_BUNDLE", clip)).resolve())


def required_assets(baseline, clip):
    baseline, clip = Path(baseline), Path(clip)
    files = [baseline / "config.json", clip / "config.json", baseline / "encoders/resnet50.pt"]
    files += [baseline / "models" / f"{kind}.pt" for kind in ("text", "image", "fusion")]
    files += [clip / "models" / f"{kind}.pt" for kind in ("clip_image", "clip_fusion")]
    for root in (baseline / "encoders/distilbert", clip / "encoders/distilbert"):
        files += [root / name for name in ("config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.txt")]
    files += [clip / "encoders/clip" / name for name in ("config.json", "model.safetensors", "preprocessor_config.json")]
    return files


def asset_signature(baseline, clip):
    files = required_assets(baseline, clip)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required trained artifacts:\n" + "\n".join(missing))
    # Small metadata key on reruns; full identity checks happen only on a cache miss.
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files)


def file_sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_evaluation(baseline, clip):
    """Use the measured five-model report, checking it belongs to these bundles."""
    baseline, clip = Path(baseline), Path(clip)
    base_config, clip_config = read_json(baseline / "config.json"), read_json(clip / "config.json")
    report = read_json(clip / "results/test_metrics.json")
    original = read_json(baseline / "results/metrics.json")
    if (report["run_id"] != clip_config["run_id"] or original["run_id"] != base_config["run_id"]
            or clip_config["baseline_run_id"] != base_config["run_id"]):
        raise ValueError("Evaluation results and loaded model runs do not match")
    if report["count"] != 500 or report["split"] != "test" or original["test_count"] != 500:
        raise ValueError("Expected the original 500-example test evaluation")
    mapped = {kind: report["models"][key] for kind, key in REPORT_KEYS.items()}
    for kind in ("text", "image", "fusion"):
        for metric, expected in original["models"][kind].items():
            actual = mapped[kind][metric]
            same = actual == expected if metric == "confusion_matrix" else math.isclose(actual, expected, abs_tol=1e-9)
            if not same:
                raise ValueError(f"Baseline report differs for {kind}/{metric}")
    return {**report, "models": mapped}
