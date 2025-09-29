from typing import Dict
import torch
import torch.nn as nn
from models.vision_backbone import VisionBackbone
from .base_probe import BaseProbe, make_head, make_head_deeper

class MultiTaskProbe(BaseProbe):
    """
    Multi-task probing condiviso:
      - backbone unica
      - una testa per task, tutte sulla stessa embedding [B, D]
    """
    def __init__(
        self,
        backbone: VisionBackbone,
        tasks: Dict[str, int],           # es: {"age": 9, "gender": 2, "emotion": 7}
        freeze_backbone: bool = True,
        dropout_p: float = 0.3,
        deeper_heads: bool = False,
        hidden_dim: int = 512,
    ):
        super().__init__(backbone, freeze_backbone)
        self.tasks = dict(tasks)
        if deeper_heads:
            self.heads = nn.ModuleDict({
                t: make_head_deeper(self.backbone.output_dim, n_cls, hidden_dim, dropout_p)
                for t, n_cls in self.tasks.items()
            })
        else:
            self.heads = nn.ModuleDict({
                t: make_head(self.backbone.output_dim, n_cls, dropout_p)
                for t, n_cls in self.tasks.items()
            })

    def forward(self, images, **kwargs):
        feats = self.extract_features(images)           # [B, D]
        feats = self._align_dtype(feats, self.heads)    # dtype -> heads
        logits = {t: self.heads[t](feats) for t in self.tasks}
        return {"logits": logits}

    @torch.no_grad()
    def predict(self, images):
        self.eval()
        out = self.forward(images)["logits"]
        return {t: logit.argmax(dim=1) for t, logit in out.items()}

    # Alias retro-compatibile
    def unfreeze_last_backbone_k_layers(self, k: int, parts: str = "all", include_embeddings: bool = True):
        super().unfreeze_last_backbone_k_layers(k=k, parts=parts, include_embeddings=include_embeddings)