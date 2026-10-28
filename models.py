"""Shared Colab/Windows model contract. Backbones are always frozen."""
from pathlib import Path
import torch
from torch import nn
from PIL import ImageOps
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel, AutoTokenizer

ENCODER_CONFIG = {"text_model": "distilbert-base-uncased", "max_length": 128,
                  "pooling": "attention_mask_mean", "image_weights": "IMAGENET1K_V2",
                  "text_dim": 768, "image_dim": 2048, "version": 1}
HEAD_CONFIG = {"text": {"dimensions": [768, 256, 2], "dropout": 0.3},
               "image": {"dimensions": [2048, 256, 2], "dropout": 0.3},
               "fusion": {"text_projection": [768, 256], "image_projection": [2048, 256],
                          "dimensions": [512, 128, 2], "dropout": 0.3}, "version": 1}


class Head(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ("text", "image", "fusion"):
            raise ValueError(kind)
        self.kind = kind
        if kind == "fusion":
            self.text = nn.Linear(768, 256)
            self.image = nn.Linear(2048, 256)
            self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.3), nn.Linear(128, 2))
        else:
            self.classifier = nn.Sequential(nn.Linear(768 if kind == "text" else 2048, 256),
                                            nn.ReLU(), nn.Dropout(.3), nn.Linear(256, 2))

    def forward(self, text, image):
        x = torch.cat((self.text(text), self.image(image)), dim=1) if self.kind == "fusion" else (text if self.kind == "text" else image)
        return self.classifier(x)


def masked_mean(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


class Encoders:
    def __init__(self, assets, device="cpu", download=False):
        self.device = torch.device(device)
        assets = Path(assets)
        text_dir = assets / "distilbert"
        source = ENCODER_CONFIG["text_model"] if download else str(text_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=not download)
        self.text = AutoModel.from_pretrained(source, local_files_only=not download)
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.image = resnet50(weights=weights if download else None)
        if download:
            assets.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(text_dir)
            self.text.save_pretrained(text_dir)
            torch.save(self.image.state_dict(), assets / "resnet50.pt")
        else:
            self.image.load_state_dict(torch.load(assets / "resnet50.pt", map_location="cpu", weights_only=True))
        self.image.fc = nn.Identity()
        self.transform = weights.transforms()
        for model in (self.text, self.image):
            model.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def encode_text(self, titles):
        tokens = self.tokenizer(titles, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        return masked_mean(self.text(**tokens).last_hidden_state, tokens["attention_mask"]).float()

    @torch.inference_mode()
    def encode_images(self, images):
        batch = torch.stack([self.transform(ImageOps.exif_transpose(im).convert("RGB")) for im in images])
        return self.image(batch.to(self.device)).float()
