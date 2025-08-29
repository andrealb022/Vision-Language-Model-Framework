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
from pathlib import Path

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
            nn.BatchNorm1d(self.backbone.output_dim),
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(self.backbone.output_dim, n_out_classes)
        )

    def extract_features(self, images):
        """
        Ritorna gli embeddings [B, D] del backbone (no_grad se freezato).
        """
        return self.backbone(images)

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

    def get_lora_target_names(self, backbone_strategy):
        """
        Ritorna i nomi dei target LoRA dalla backbone e dalla head.
        """
        # (A) target dalla backbone
        bk_rel = self.backbone.get_lora_target_names(backbone_strategy)   # es: ['vision_model.encoder.layers.23.self_attn.q_proj', ...]
        bk_full = [f"backbone.{n}" for n in bk_rel]

        # (B) target dalla testa: tutti i Linear sotto 'classifier'
        head = [n for n, m in self.named_modules() if n.startswith("classifier.") and isinstance(m, nn.Linear)]
        # (C) unisci
        target_modules = sorted(set(bk_full + head))
        return target_modules