from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests

from PIL import Image, ImageOps
from tqdm import tqdm


DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"

SPLITS = {
    "train": ("multimodal_train.tsv", 3000),
    "validate": ("multimodal_validate.tsv", 500),
    "test": ("multimodal_test_public.tsv", 500),
}

IMAGE_SIZE = (224, 224)
MAX_WORKERS = 8


def download_image(row, split):
    image_id = str(row["id"])
    image_url = str(row["image_url"])

    output_dir = IMAGE_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_id}.jpg"

    if output_path.exists():
        try:
            with Image.open(output_path) as existing:
                existing.verify()
            return str(output_path)
        except Exception:
            output_path.unlink(missing_ok=True)

    for attempt in range(3):
        try:
            response = requests.get(
                image_url,
                timeout=12,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            image.thumbnail(IMAGE_SIZE)

            canvas = Image.new("RGB", IMAGE_SIZE, (0, 0, 0))

            x = (IMAGE_SIZE[0] - image.width) // 2
            y = (IMAGE_SIZE[1] - image.height) // 2

            canvas.paste(image, (x, y))

            canvas.save(
                output_path,
                format="JPEG",
                quality=85,
            )

            return str(output_path)

        except Exception:
            output_path.unlink(missing_ok=True)

    return None


def process_split(split, filename, target_count):
    print(f"\nProcessing {split} dataset")

    df = pd.read_csv(
        DATA_DIR / filename,
        sep="\t",
        low_memory=False,
    )

    required_columns = [
        "id",
        "clean_title",
        "image_url",
        "2_way_label",
    ]

    df = df.dropna(subset=required_columns)

    df = df[
        df["clean_title"].astype(str).str.strip().ne("")
        & df["image_url"].astype(str).str.startswith(
            ("http://", "https://")
        )
    ]

    df = df.drop_duplicates(subset=["id"])

    df = df.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    selected_rows = []

    batch_size = 500

    with tqdm(total=target_count, desc=split) as progress:

        for start in range(0, len(df), batch_size):

            if len(selected_rows) >= target_count:
                break

            batch = df.iloc[start:start + batch_size]

            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                futures = {
                    executor.submit(
                        download_image,
                        row,
                        split
                    ): row
                    for _, row in batch.iterrows()
                }

                for future in as_completed(futures):

                    image_path = future.result()

                    if image_path is None:
                        continue

                    if len(selected_rows) >= target_count:
                        continue

                    row = futures[future].copy()
                    row["local_image_path"] = image_path

                    selected_rows.append(row)

                    progress.update(1)

    if not selected_rows:
        raise RuntimeError(
            f"No usable image-text pairs found for {split}"
        )

    result = pd.DataFrame(selected_rows)

    output_file = DATA_DIR / f"{split}_processed.csv"

    result.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved {len(result)} samples to {output_file}")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    for split, (filename, count) in SPLITS.items():
        process_split(
            split,
            filename,
            count,
        )

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()