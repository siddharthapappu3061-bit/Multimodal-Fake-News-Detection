"""Read existing paired data without rewriting CSVs or downloading images."""
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
from PIL import Image

SIZES = {"train": 3000, "validate": 500, "test": 500}
LABELS = {0: "Fake", 1: "Real"}


def resolve_image(root, raw):
    root = Path(root).resolve()
    parts = str(raw).replace("\\", "/").split("/")
    # Relocate even absolute Windows paths by the images/ suffix.
    if "images" not in parts:
        raise ValueError(f"Image path has no images directory: {raw}")
    relative = Path(*parts[parts.index("images"):])
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Unsafe image path: {raw}")
    return target


def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            name = entry.filename.replace("\\", "/")
            target = (destination / name).resolve()
            if not target.is_relative_to(destination) or ":" in name:
                raise ValueError(f"Unsafe ZIP member: {name}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are not supported")
        for entry in z.infolist():
            target = destination / entry.filename.replace("\\", "/")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(entry) as src, target.open("wb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
    matches = list(destination.rglob("train_processed.csv"))
    if len(matches) != 1:
        raise ValueError("ZIP must contain exactly one processed dataset")
    return matches[0].parent


def validate(root, report_path):
    frames, report, errors = {}, {"label_mapping": LABELS, "splits": {}}, []
    for split, expected in SIZES.items():
        df = pd.read_csv(Path(root) / f"{split}_processed.csv", dtype={"id": str}, keep_default_na=False)
        required = ["id", "clean_title", "2_way_label", "local_image_path"]
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns in {split}: {set(required) - set(df.columns)}")
        missing = {c: int(df[c].astype(str).str.strip().eq("").sum()) for c in required}
        duplicates = df.loc[df.id.duplicated(), "id"].tolist()
        if len(df) != expected or any(missing.values()) or duplicates:
            errors.append(f"{split}: count/missing/duplicate failure")
        numeric = pd.to_numeric(df["2_way_label"], errors="coerce")
        if not numeric.isin([0, 1]).all():
            errors.append(f"{split}: labels must be 0 or 1")
        if "6_way_label" in df:
            six = pd.to_numeric(df["6_way_label"], errors="coerce")
            if not six.isin(range(6)).all() or not ((six == 0) == (numeric == 1)).all():
                errors.append(f"{split}: binary labels inconsistent with 6-way true category")
        paths, hashes, bad = [], [], []
        for row in df.itertuples(index=False):
            try:
                p = resolve_image(root, row.local_image_path)
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.convert("RGB").load()
                hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
                paths.append(str(p))
            except Exception as exc:
                bad.append({"id": row.id, "error": str(exc)})
                paths.append("")
                hashes.append("")
        if bad:
            errors.append(f"{split}: {len(bad)} missing/corrupt images")
        df["resolved_path"], df["image_sha256"] = paths, hashes
        frames[split] = df
        report["splits"][split] = {"count": len(df), "classes": df["2_way_label"].value_counts().to_dict(),
                                  "missing": missing, "duplicate_ids": duplicates, "bad_images": bad}
    names = list(frames)
    report["overlaps"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sorted(set(frames[a].id) & set(frames[b].id))
            report["overlaps"][f"{a}/{b}"] = overlap
            if overlap:
                errors.append(f"{a}/{b}: overlapping sample IDs")
    report["errors"] = errors
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise ValueError(f"Validation failed; see {report_path}: {errors}")
    return frames, report


if __name__ == "__main__":
    _, report = validate("data", "results/dataset_validation.json")
    print(json.dumps(report, indent=2))
