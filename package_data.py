"""Package only existing processed CSVs and their referenced images. No downloads."""
from pathlib import Path
import zipfile
import pandas as pd
from dataset import resolve_image, SIZES


def package(root="data", output="fakeddit_processed.zip"):
    root = Path(root)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        seen = set()
        for split in SIZES:
            csv = root / f"{split}_processed.csv"
            z.write(csv, f"data/{csv.name}")
            for raw in pd.read_csv(csv).local_image_path:
                image = resolve_image(root, raw)
                name = "data/" + image.relative_to(root.resolve()).as_posix()
                if name not in seen:
                    z.write(image, name)
                    seen.add(name)
    print(f"Created {output} with {len(seen)} existing images; no original TSVs or virtual environment included.")


if __name__ == "__main__":
    package()
