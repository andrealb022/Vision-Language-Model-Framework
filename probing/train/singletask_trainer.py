from pathlib import Path
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from probing.train.base_trainer import BaseTrainer
from probing.train.utils import (
    collate_keep_pil,
    get_num_classes_for_task,
    counts_to_weights,
    targets_to_tensors,
)
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from probing.models.linear_probe import LinearProbe
from tqdm import tqdm


class SingleTaskTrainer(BaseTrainer):
    """
    Single-task trainer semplificato:
      - Il bilanciamento avviene solo via class weights nella CrossEntropy.
      - Pre-estrazione + caching feature quando la backbone è freeze (default ON).
    """

    def __init__(self, cfg: dict, run_name: str, ckpt_root: Path):
        self.task = str(cfg["task"]).lower()
        self.use_feature_cache = False
        self.features_dir: Optional[Path] = None
        self.head_only: Optional[nn.Module] = None
        super().__init__(cfg, run_name, ckpt_root)

    # ------------ MODEL ------------
    def build_model(self) -> nn.Module:
        mcfg = self.cfg["model"]
        bb_cfg = (mcfg.get("backbone") or {})
        # Backward-compat con vecchio campo
        freeze_flag = bool(bb_cfg.get("freeze", True))
        unfreeze_k = int(bb_cfg.get("unfreeze_last_k", 0))
        unfreeze_parts = str(bb_cfg.get("unfreeze_parts", "all"))
        include_embeddings = bool(bb_cfg.get("include_embeddings", True))

        vlm = VLMModelFactory.create_model(
            mcfg["name"], model_id=None, device=torch.device("cpu"), quantization=mcfg.get("quantization")
        )
        backbone = vlm.get_vision_backbone(); del vlm

        probe = LinearProbe(
            backbone=backbone,
            n_out_classes=get_num_classes_for_task(self.task),
            freeze_backbone=freeze_flag,
            dropout_p=float(mcfg.get("dropout_p", 0.3)),
            deeper_head=bool(mcfg.get("deeper_head", False)),
            hidden_dim=int(mcfg.get("hidden_dim", 512)),
        )

        # Se congelata ma con k>0, sblocca solo gli ultimi k layer
        if freeze_flag and unfreeze_k > 0:
            probe.unfreeze_last_backbone_k_layers(
                k=unfreeze_k, parts=unfreeze_parts, include_embeddings=include_embeddings
            )

        return probe

    # ------------ DATALOADERS ------------
    def build_dataloaders(self):
        dcfg = self.cfg["data"]
        base_path   = dcfg.get("base_path", None)
        batch_size  = int(dcfg.get("batch_size", 64))
        num_workers = int(dcfg.get("num_workers", 8))
        nclasses = {self.task: get_num_classes_for_task(self.task)}

        # dataset immagine (ordine deterministico per estrazione feature)
        train_img_ds, agg_counts = DatasetFactory.create_task_dataset(
            tasks=[self.task], split="train", base_path=base_path, transform=None, num_classes=nclasses
        )
        val_img_ds, _ = DatasetFactory.create_task_dataset(
            tasks=[self.task], split="val", base_path=base_path, transform=None, num_classes=nclasses
        )

        # --- Class weights per la loss (unico bilanciamento) ---
        counts = agg_counts.get(self.task) if isinstance(agg_counts, dict) else None
        if counts is None:
            w = np.ones(get_num_classes_for_task(self.task), dtype=np.float64)
        else:
            w = counts_to_weights(np.asarray(counts, dtype=np.float64))
        self.class_weights = {
            self.task: torch.tensor(w, dtype=torch.float32)
        }
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights[self.task].to(self.device),
            ignore_index=-1
        ).to(self.device)

        mcfg = self.cfg["model"]
        # --- Feature cache: default TRUE solo se backbone è davvero tutta frozen ---
        fully_frozen = not any(p.requires_grad for p in self.model.backbone.parameters())
        self.use_feature_cache = fully_frozen # default: usa cache solo se fully frozen
        print(f"[Trainer] Feature cache for probing: {'ENABLED' if self.use_feature_cache else 'DISABLED'} (backbone fully frozen: {fully_frozen})")

        if self.use_feature_cache:
            project_root = Path(os.getenv("PYTHONPATH") or ".")
            model_name   = mcfg["name"]
            quant        = mcfg.get("quantization")
            # path fisso richiesto
            self.features_dir = project_root / "probing" / "linear_probing" / "features" / f"{model_name}_{quant}_{self.task}"
            self.features_dir.mkdir(parents=True, exist_ok=True)

            # prepara (estrae se mancante) per train/val
            train_feat_ds = self._ensure_features(train_img_ds, split="train")
            val_feat_ds   = self._ensure_features(val_img_ds,   split="val")

            # DataLoader senza sampler (solo shuffle per il train)
            train_loader = DataLoader(
                train_feat_ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=False,
            )
            val_loader   = DataLoader(
                val_feat_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, drop_last=False,
            )

            # head-only: la head del probe prende le feature e restituisce i logits
            self.head_only = self.model.classifier.to(self.device).train()
            return train_loader, val_loader

        # ---- fallback: training end-to-end (niente cache, niente sampler) ----
        train_loader = DataLoader(
            train_img_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_keep_pil, persistent_workers=(num_workers > 0)
        )
        val_loader   = DataLoader(
            val_img_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_keep_pil, persistent_workers=(num_workers > 0)
        )
        self.head_only = None
        return train_loader, val_loader

    def post_build(self):
        tcfg = self.cfg.get("train", {})
        head_lr = float(tcfg.get("lr", 1e-4))
        backbone_lr = float(tcfg.get("backbone_lr", head_lr))
        weight_decay = float(tcfg.get("weight_decay", 1e-4))

        if self.use_feature_cache:
            # Alleni solo la head (self.head_only)
            params = list(self.head_only.parameters())
            self.optimizer = torch.optim.AdamW(
                [{"params": params, "lr": head_lr}],
                lr=head_lr, weight_decay=weight_decay
            )
        else:
            # End-to-end: separa heads vs backbone
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
            head_params = [p for n, p in self.model.named_parameters()
                        if p.requires_grad and not n.startswith("backbone.")]
            groups = []
            if head_params:
                groups.append({"params": head_params, "lr": head_lr})
            if backbone_params:
                groups.append({"params": backbone_params, "lr": backbone_lr})

            self.optimizer = torch.optim.AdamW(groups, lr=head_lr, weight_decay=weight_decay)

        scfg = (tcfg.get("scheduler") or {"type": "cosine_wr", "T_0": 10, "T_mult": 2})
        if (scfg or {}).get("type", "cosine_wr") == "cosine_wr":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=int(scfg.get("T_0", 10)), T_mult=int(scfg.get("T_mult", 2))
            )
        else:
            self.scheduler = None

    # ------------ LOSS STEP ------------
    def compute_losses(self, batch, train: bool = True) -> dict:
        if self.use_feature_cache:
            # batch = (features, y)
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.head_only(x)  # solo head
            loss_t = self.criterion(logits, y)
            return {self.task: loss_t}

        # end-to-end fallback
        images_list, targets_list = batch
        y = targets_to_tensors(targets_list, [self.task], device=self.device)[self.task]
        logits = self.model(images_list)  # [B, C]
        loss_t = self.criterion(logits, y)
        return {self.task: loss_t}

    def run_meta(self) -> dict:
        meta = super().run_meta()
        mcfg = self.cfg["model"]
        bb_cfg = (mcfg.get("backbone") or {})
        meta.update({
            "trainer": "single_task",
            "task": self.task,
            "feature_cache": bool(self.use_feature_cache),
            "sampler": "none",
            "backbone": {
                "freeze": bool(bb_cfg.get("freeze", mcfg.get("freeze_backbone", True))),
                "unfreeze_last_k": int(bb_cfg.get("unfreeze_last_k", 0)),
                "unfreeze_parts": str(bb_cfg.get("unfreeze_parts", "all")),
                "include_embeddings": bool(bb_cfg.get("include_embeddings", True)),
            },
        })
        return meta

    # ------------ FEATURE CACHE UTILS ------------
    @torch.no_grad()
    def _ensure_features(
        self,
        img_dataset,
        split: str,
    ) -> TensorDataset:
        """
        Ritorna un TensorDataset(features, y) per lo split richiesto.
        Se esiste una cache su disco, la usa. Altrimenti estrae le feature
        chiamando direttamente `self.model.extract_features(images)` con una tqdm.
        """
        assert self.features_dir is not None
        fpath = self.features_dir / f"{split}_features.pt"

        # --- Cache già presente ---
        if fpath.exists():
            blob = torch.load(fpath, map_location="cpu")
            x_key = "x" if "x" in blob else ("features" if "features" in blob else ("feats" if "feats" in blob else None))
            y_key = "y" if "y" in blob else ("labels" if "labels" in blob else None)
            if x_key is None or y_key is None:
                raise KeyError(f"Cache feature non riconosciuta: chiavi presenti {list(blob.keys())}")
            feats = blob[x_key].contiguous()
            ys    = blob[y_key].long().contiguous()
            return TensorDataset(feats, ys)

        # --- Estrazione features ---
        self.model.eval().to(self.device)

        loader = DataLoader(
            img_dataset,
            batch_size=int(self.cfg["data"].get("batch_size", 64)),
            shuffle=False,
            num_workers=int(self.cfg["data"].get("num_workers", 8)),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_keep_pil,
            persistent_workers=False,
        )

        feats_all: List[torch.Tensor] = []
        ys_all:    List[torch.Tensor] = []

        use_amp = (self.device.type == "cuda")
        pbar = tqdm(loader, desc=f"Estrazione features [{split}]", dynamic_ncols=True)

        # NB: extract_features gestisce già no_grad se freeze_backbone=True
        for images_list, targets_list in pbar:
            with torch.autocast(device_type=('cuda' if use_amp else 'cpu'),
                                dtype=self.autocast_dtype,
                                enabled=use_amp):
                feats_b = self.model.extract_features(images_list)   # -> [B, D] sul device
            feats_all.append(feats_b.detach().cpu())

            y_b = targets_to_tensors(targets_list, [self.task], device=None)[self.task]  # cpu
            ys_all.append(y_b)

        self.model.train()  # ripristina lo stato del trainer

        feats = torch.cat(feats_all, dim=0).contiguous()
        ys    = torch.cat(ys_all,   dim=0).long().contiguous()

        # salva cache
        blob = {"x": feats, "y": ys}
        fpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, fpath)

        return TensorDataset(feats, ys)