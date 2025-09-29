from typing import Optional
import torch
import torch.nn as nn
from models.vision_backbone import VisionBackbone
from .base_probe import BaseProbe, make_head, make_head_deeper

class LinearProbe(BaseProbe):
    """
    Linear probing: backbone (freeze opzionale) + testa (lineare o "deeper").
    Compatibile con i tuoi trainer:
      - forward(images) -> logits [B, C]
      - extract_features(images) usata per il caching offline
    """
    def __init__(
        self,
        backbone: VisionBackbone,
        n_out_classes: int,
        freeze_backbone: bool = True,
        dropout_p: float = 0.3,
        deeper_head: bool = False,
        hidden_dim: int = 512,
    ):
        super().__init__(backbone, freeze_backbone)
        if deeper_head:
            self.classifier = make_head_deeper(
                in_dim=self.backbone.output_dim, out_dim=n_out_classes,
                hidden_dim=hidden_dim, dropout_p=dropout_p
            )
        else:
            self.classifier = make_head(
                in_dim=self.backbone.output_dim, out_dim=n_out_classes,
                dropout_p=dropout_p
            )

    def forward(self, images, **kwargs):
        feats = self.extract_features(images)             # [B, D]
        feats = self._align_dtype(feats, self.classifier) # dtype -> head
        logits = self.classifier(feats)                   # [B, C]
        return logits

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        logits = self.forward(images)
        return torch.argmax(logits, dim=-1)

    # Alias retro-compatibile
    def unfreeze_last_backbone_k_layers(self, k: int, parts: str = "all", include_embeddings: bool = True):
        super().unfreeze_last_backbone_k_layers(k=k, parts=parts, include_embeddings=include_embeddings)