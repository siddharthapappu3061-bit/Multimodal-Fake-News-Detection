"""Offline CPU inference from a Colab-exported bundle."""
import json
from pathlib import Path
import torch
from models import Encoders, Head, ENCODER_CONFIG, HEAD_CONFIG


class Predictor:
    def __init__(self, bundle="artifacts"):
        bundle = Path(bundle)
        config = json.loads((bundle / "config.json").read_text())
        if (config["encoders"] != ENCODER_CONFIG or config["heads"] != HEAD_CONFIG
                or config["labels"] != {"0": "Fake", "1": "Real"}):
            raise ValueError("Incompatible model configuration")
        self.encoders = Encoders(bundle / "encoders", "cpu")
        self.heads = {}
        for kind in ("text", "image", "fusion"):
            checkpoint = torch.load(bundle / "models" / f"{kind}.pt", map_location="cpu", weights_only=True)
            if checkpoint["kind"] != kind or checkpoint["run_id"] != config["run_id"]:
                raise ValueError("Mixed checkpoint bundle")
            model = Head(kind)
            model.load_state_dict(checkpoint["state_dict"])
            self.heads[kind] = model.eval()

    @torch.inference_mode()
    def predict(self, title, image):
        if not title.strip():
            raise ValueError("Enter a headline")
        text = self.encoders.encode_text([title.strip()])
        visual = self.encoders.encode_images([image])
        return {kind: model(text, visual).softmax(1)[0].tolist() for kind, model in self.heads.items()}
