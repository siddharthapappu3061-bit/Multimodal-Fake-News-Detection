"""Experimental CLIP heads and offline encoders; independent of the production app."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from PIL import ImageOps
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

CLIP_ID = "openai/clip-vit-base-patch32"
FEATURE_SPEC = {"model": CLIP_ID, "output": "image_embeds", "projection_dim": 512,
                "l2_normalize": True, "processor": "CLIPImageProcessor", "version": 1}


class ClipHead(nn.Module):
    def __init__(self, kind, text_dim=768, image_dim=512, dropout=0.3):
        super().__init__()
        if kind not in ("clip_image", "clip_fusion"):
            raise ValueError(f"Unknown experimental head: {kind}")
        if text_dim <= 0 or image_dim <= 0:
            raise ValueError("Feature dimensions must be positive")
        self.spec = dict(kind=kind, text_dim=text_dim, image_dim=image_dim, dropout=dropout)
        self.kind = kind
        if kind == "clip_fusion":
            self.text_projection = nn.Linear(text_dim, 256)
            self.image_projection = nn.Linear(image_dim, 256)
            self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))
        else:
            self.classifier = nn.Sequential(nn.Linear(image_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 2))

    def forward(self, text, image):
        if image.ndim != 2 or image.shape[1] != self.spec["image_dim"]:
            raise ValueError("Wrong CLIP image feature dimension")
        if self.kind == "clip_fusion":
            if text.ndim != 2 or text.shape != (len(image), self.spec["text_dim"]):
                raise ValueError("Wrong text feature dimension or pairing")
            image = torch.cat([self.text_projection(text), self.image_projection(image)], dim=1)
        return self.classifier(image)


class ClipEncoder:
    def __init__(self, source, device="cpu", revision=None, local=True):
        kwargs = {"local_files_only": local}
        if revision is not None:
            kwargs["revision"] = revision
        self.processor = CLIPImageProcessor.from_pretrained(str(source), **kwargs)
        self.model = CLIPVisionModelWithProjection.from_pretrained(str(source), **kwargs)
        self.model.requires_grad_(False).eval().to(device)
        self.device = torch.device(device)
        self.dimension = self.model.config.projection_dim
        if self.dimension != FEATURE_SPEC["projection_dim"]:
            raise ValueError("Unexpected CLIP projection size")

    def save(self, path):
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)

    @torch.inference_mode()
    def encode(self, images):
        images = [ImageOps.exif_transpose(im).convert("RGB") for im in images]
        inputs = self.processor(images=images, return_tensors="pt")
        image = self.model(pixel_values=inputs.pixel_values.to(self.device)).image_embeds.float()
        return F.normalize(image, p=2, dim=-1)


class CachedTextEncoder:
    """Use exactly the baseline's exported DistilBERT and masked-mean preprocessing."""
    def __init__(self, source, device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(source), local_files_only=True)
        self.model.requires_grad_(False).eval().to(device)
        self.device = torch.device(device)

    @torch.inference_mode()
    def encode(self, titles):
        tokens = self.tokenizer(titles, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float()
