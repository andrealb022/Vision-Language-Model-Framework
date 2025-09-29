from typing import Dict
import torch
import torch.nn as nn
from models.vision_backbone import VisionBackbone

# ------------------------ helper per le heads ------------------------
def make_head(in_dim: int, out_dim: int, dropout_p: float) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, out_dim),
    )

def make_head_deeper(in_dim: int, out_dim: int, hidden_dim: int, dropout_p: float) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(hidden_dim, out_dim),
    )

# ------------------------ classe base ------------------------
class BaseProbe(nn.Module):
    """
    Logica comune per i probe:
      - gestione backbone + freeze/unfreeze
      - estrazione feature con no_grad se completamente frozen
      - allineamento dtype delle feature verso la/e testa/e
    """
    def __init__(self, backbone: VisionBackbone, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        
        # Applica freeze globale se richiesto
        self.set_freeze_backbone(bool(freeze_backbone))
        self._fully_frozen = self._compute_fully_frozen()

    # ---- utilità interne ----
    def _compute_fully_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.backbone.parameters())

    @staticmethod
    def _align_dtype(x: torch.Tensor, ref_module: nn.Module) -> torch.Tensor:
        try:
            target_dtype = next(ref_module.parameters()).dtype
        except StopIteration:
            return x
        return x if x.dtype == target_dtype else x.to(target_dtype)

    # ---- API comuni ----
    def extract_features(self, images):
        """
        Se la backbone è di fatto completamente congelata -> eval() + no_grad().
        Altrimenti esegue il forward con grad abilitato (per layer sbloccati).
        """
        if self._fully_frozen:
            was_training = self.backbone.training
            self.backbone.eval()
            with torch.no_grad():
                feats = self.backbone(images)
            if was_training:
                self.backbone.train()
            return feats
        else:
            return self.backbone(images)

    def unfreeze_last_backbone_k_layers(self, k: int, parts: str = "all", include_embeddings: bool = True):
        """
        Sblocca selettivamente gli ultimi k layer del vision encoder (delegato alla backbone).
        Aggiorna la cache di freeze.
        """
        self.backbone.unfreeze_last_k_layers(k=k, parts=parts, include_embeddings=include_embeddings)
        self._fully_frozen = self._compute_fully_frozen()

    def set_freeze_backbone(self, freeze: bool):
        """
        Applica freeze/unfreeze globale alla backbone e aggiorna la cache.
        """
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        self._fully_frozen = self._compute_fully_frozen()