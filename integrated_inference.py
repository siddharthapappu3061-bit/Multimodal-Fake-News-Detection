"""Five trained classifiers on CPU with one shared DistilBERT encoder."""
from pathlib import Path
from threading import RLock

import torch

from inference import Predictor
from models import ENCODER_CONFIG
from clip_models import ClipEncoder, ClipHead, FEATURE_SPEC
from integration_support import asset_signature, file_sha256, read_json


class FiveModelPredictor:
    def __init__(self, baseline_bundle, clip_bundle):
        baseline_bundle, clip_bundle = Path(baseline_bundle), Path(clip_bundle)
        asset_signature(baseline_bundle, clip_bundle)  # List every missing artifact before loading.
        base_config = read_json(baseline_bundle / "config.json")
        clip_config = read_json(clip_bundle / "config.json")
        if (clip_config["labels"] != {"0": "Fake", "1": "Real"}
                or clip_config["feature_spec"] != FEATURE_SPEC
                or clip_config["text_encoder"] != ENCODER_CONFIG
                or clip_config["baseline_run_id"] != base_config["run_id"]):
            raise ValueError("CLIP bundle is incompatible with the selected baseline")
        # Prove text weights/tokenization are identical before sharing the encoder.
        source_hashes = clip_config["provenance"]["baseline"]["text_asset_sha256"]
        required_names = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"}
        if not required_names.issubset(source_hashes):
            raise ValueError("CLIP export lacks complete DistilBERT provenance")
        for name, expected in source_hashes.items():
            if Path(name).name != name:
                raise ValueError("Unexpected text asset path in provenance")
            for bundle in (baseline_bundle, clip_bundle):
                path = bundle / "encoders/distilbert" / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing text encoder asset: {path}")
                if file_sha256(path) != expected:
                    raise ValueError(f"DistilBERT assets differ from trained CLIP provenance: {path}")
        for kind, expected in clip_config["provenance"]["baseline"]["checkpoint_sha256"].items():
            if kind not in ("text", "image", "fusion"):
                raise ValueError("Unexpected baseline head in provenance")
            if file_sha256(baseline_bundle / "models" / f"{kind}.pt") != expected:
                raise ValueError(f"Baseline checkpoint has changed: {kind}")
        # Avoid excessive thread overhead for single-example desktop inference.
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)
        self.baseline = Predictor(baseline_bundle)
        self.clip = ClipEncoder(clip_bundle / "encoders/clip", device="cpu", local=True)
        self.clip_heads = {}
        for kind in ("clip_image", "clip_fusion"):
            saved = torch.load(clip_bundle / "models" / f"{kind}.pt", map_location="cpu", weights_only=True)
            spec = saved["spec"]
            if (saved["run_id"] != clip_config["run_id"] or spec != clip_config["heads"][kind]
                    or spec["kind"] != kind or spec["text_dim"] != 768 or spec["image_dim"] != self.clip.dimension):
                raise ValueError(f"Invalid CLIP checkpoint configuration: {kind}")
            head = ClipHead(**spec)
            head.load_state_dict(saved["state_dict"], strict=True)
            self.clip_heads[kind] = head.requires_grad_(False).eval()
        for head in self.baseline.heads.values():
            head.requires_grad_(False)
        self.run_ids = (base_config["run_id"], clip_config["run_id"])
        self._lock = RLock()

    @torch.inference_mode()
    def predict(self, title, image):
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Enter a headline")
        # Original modules retain their own respective image preprocessing.
        with self._lock:
            text = self.baseline.encoders.encode_text([title.strip()])
            resnet = self.baseline.encoders.encode_images([image])
            clip = self.clip.encode([image])
            logits = {kind: model(text, resnet) for kind, model in self.baseline.heads.items()}
            logits.update({kind: model(text, clip) for kind, model in self.clip_heads.items()})
            result = {}
            for kind, value in logits.items():
                if value.shape != (1, 2) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid classifier output: {kind}")
                result[kind] = value.softmax(1)[0].cpu().tolist()
            return result
