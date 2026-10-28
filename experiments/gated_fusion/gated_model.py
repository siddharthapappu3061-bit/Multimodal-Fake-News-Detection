"""Differentiable gated fusion over existing frozen-encoder features."""
import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, text_dim, image_dim, branch_dropout=0.2, head_dropout=0.3):
        super().__init__()
        if text_dim < 1 or image_dim < 1:
            raise ValueError("Use positive dimensions measured from verified cached tensors")
        self.spec = dict(text_dim=text_dim, image_dim=image_dim,
                         branch_dropout=branch_dropout, head_dropout=head_dropout)
        def branch(dimension):
            return nn.Sequential(nn.Linear(dimension, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(branch_dropout))
        self.text_branch = branch(text_dim)
        self.image_branch = branch(image_dim)
        self.gate = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(head_dropout), nn.Linear(128, 2))

    def forward(self, text, image):
        if text.ndim != 2 or image.ndim != 2 or text.shape != (len(image), self.spec['text_dim']) or image.shape[1] != self.spec['image_dim']:
            raise ValueError("Wrong feature dimensions or unpaired batches")
        text, image = self.text_branch(text), self.image_branch(image)
        gates = self.gate(torch.cat([text, image], dim=1))
        fused = torch.cat([gates[:, :1] * text, gates[:, 1:] * image], dim=1)
        return self.classifier(fused), gates[:, 0], gates[:, 1]
