"""CPU-only loader for a separate clip_artifacts bundle. Does not touch Streamlit."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from clip_models import ClipEncoder, ClipHead, CachedTextEncoder, FEATURE_SPEC


class ClipPredictor:
    def __init__(self, bundle="clip_artifacts"):
        bundle = Path(bundle)
        self.config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        if self.config["labels"] != {"0": "Fake", "1": "Real"} or self.config["feature_spec"] != FEATURE_SPEC:
            raise ValueError("Incompatible experimental bundle")
        self.clip = ClipEncoder(bundle / "encoders/clip", device="cpu", local=True)
        self.text = CachedTextEncoder(bundle / "encoders/distilbert", device="cpu")
        self.heads = {}
        for kind in ("clip_image", "clip_fusion"):
            saved = torch.load(bundle / "models" / f"{kind}.pt", map_location="cpu", weights_only=True)
            if saved["run_id"] != self.config["run_id"] or saved["spec"] != self.config["heads"][kind]:
                raise ValueError("Mixed experiment checkpoints")
            model = ClipHead(**saved["spec"]).eval()
            model.load_state_dict(saved["state_dict"])
            self.heads[kind] = model

    @torch.inference_mode()
    def predict(self, title, image):
        if not title.strip():
            raise ValueError("Enter nonempty text for fusion")
        text = self.text.encode([title.strip()])
        visual = self.clip.encode([image])
        return {kind: model(text, visual).softmax(1)[0].tolist() for kind, model in self.heads.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="clip_artifacts")
    parser.add_argument("--text", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    with Image.open(args.image) as image:
        print(json.dumps(ClipPredictor(args.bundle).predict(args.text, image), indent=2))
