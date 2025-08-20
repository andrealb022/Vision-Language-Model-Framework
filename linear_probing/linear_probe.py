# Percorso di progetto
from dotenv import load_dotenv
import os, sys
load_dotenv()   # carica variabili da .env
project_root = os.getenv("PYTHONPATH")  # aggiungi PYTHONPATH se definito
if project_root and project_root not in sys.path:
    sys.path.append(project_root)
import torch
import torch.nn as nn
from models.base_vision_backbone import VisionBackbone

# Variabili
DROPOUT_P = 0.3

class LinearProbe(nn.Module):
    """
    Linear probing: backbone (freezato di default) + testa lineare.
    """
    def __init__(self, backbone: VisionBackbone, n_out_classes: int, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(self.backbone.output_dim, n_out_classes)
        )

    def extract_features(self, images):
        """
        Ritorna gli embeddings [B, D] del backbone (no_grad se freezato).
        """
        if all(not p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)

    def forward(self, images):
        feats = self.extract_features(images)   # [B, D]

        # Casting delle feature al dtype del classifier
        target_dtype = next(self.classifier.parameters()).dtype
        if feats.dtype != target_dtype:
            feats = feats.to(target_dtype)
        logits = self.classifier(feats)         # [B, C]
        return logits

    def save(self):
        pass
    
    def load(self):
        pass
    
    def save_head(self):
        pass