from dotenv import load_dotenv
import os, sys
load_dotenv()   # carica variabili da .env
project_root = os.getenv("PYTHONPATH")  # aggiungi PYTHONPATH se definito
if project_root and project_root not in sys.path:
    sys.path.append(project_root)
import torch
import torch.nn as nn
from models.vision_backbone import VisionBackbone
from pathlib import Path

def _make_head(in_dim: int, out_dim: int, dropout_p: float) -> nn.Sequential:
    # identica filosofia al LinearProbe: BN1d -> Dropout -> Linear
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, out_dim)
    )

def _make_head_deeper(in_dim: int, out_dim: int, hidden_dim: int, dropout_p: float) -> nn.Sequential:
    # variante MLP leggera: BN1d -> Dropout -> Linear -> GELU -> Dropout -> Linear
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(hidden_dim, out_dim)
    )

class LinearProbe(nn.Module):
    """
    Linear probing: backbone (freezato di default) + testa lineare.
    """
    def __init__(self, backbone: VisionBackbone, n_out_classes: int, freeze_backbone: bool = True, dropout_p: float = 0.3, 
    deeper_head: bool = False, hidden_dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if deeper_head:
            self.classifier = _make_head_deeper(in_dim=self.backbone.output_dim, out_dim=n_out_classes, 
            hidden_dim=hidden_dim, dropout_p=dropout_p)
        else:
            self.classifier = _make_head(in_dim=self.backbone.output_dim, out_dim=n_out_classes, dropout_p=dropout_p)

    def extract_features(self, images):
        """
        Ritorna l'embedding condivisa [B, D] del backbone.
        """
        was_training = self.backbone.training
        self.backbone.eval()  # per sicurezza con BN/Dropout interni al backbone

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.backbone(images)
        else:
            feats = self.backbone(images)

        if was_training:
            self.backbone.train()
        return feats

    def forward(self, images, **kwargs):
        """
        Metodo di forward del modello.
        Args:
            images: PIL.Image o List[PIL.Image] (il formato esatto può dipendere dal processor).
        """
        feats = self.extract_features(images)   # [B, D]

        # Casting delle feature al dtype del classifier
        target_dtype = next(self.classifier.parameters()).dtype
        if feats.dtype != target_dtype:
            feats = feats.to(target_dtype)
        logits = self.classifier(feats)         # [B, C]
        return logits

    @torch.no_grad()
    def predict(self, images):
        """
        Ritorna le predizioni argmax per ogni task.
        """
        self.eval()
        logits = self.forward(images)           # [B, C]
        preds = torch.argmax(logits, dim=-1)    # [B]
        return preds