from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from probing.train.base_trainer import BaseTrainer
from probing.train.utils import (
    collate_keep_pil,
    get_num_classes_for_task,
    counts_to_weights,
    targets_to_tensors,
    build_weighted_sampler,
)
from probing.train.losses import RunningMeans
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from probing.models.multitask_probe import MultiTaskProbe
import torchvision.transforms as transforms

class MultiTaskTrainer(BaseTrainer):
    def __init__(self, cfg: dict, run_name: str, ckpt_root: Path):
        self.tasks = [t.lower() for t in cfg["tasks"]]
        # RunningMeans config
        tcfg = cfg["train"]
        rm_cfg = (tcfg.get("running_means") or {})
        self.use_running_means = bool(rm_cfg.get("enabled", True))
        self.rm_alpha = float(rm_cfg.get("alpha", 0.95))
        self.rm: Optional[RunningMeans] = None
        
        # pesi statici (fallback per prime epoche / EMA non inizializzata)
        tw_cfg = (tcfg.get("task_weights") or {})
        self.static_task_weights = {t: float(tw_cfg.get(t, 1.0)) for t in self.tasks}
        self.current_task_weights = {t: 1.0 for t in self.tasks}

        super().__init__(cfg, run_name, ckpt_root)

    def build_model(self) -> nn.Module:
        mcfg = self.cfg["model"]
        bb_cfg = (mcfg.get("backbone") or {})
        freeze_backbone_flag = bool(bb_cfg.get("freeze", True))
        unfreeze_k = int(bb_cfg.get("unfreeze_last_k", 0))
        unfreeze_parts = str(bb_cfg.get("unfreeze_parts", "all"))
        include_embeddings = bool(bb_cfg.get("include_embeddings", True))

        vlm = VLMModelFactory.create_model(
            mcfg["name"], model_id=None, device=torch.device("cpu"), quantization=mcfg.get("quantization")
        )
        backbone = vlm.get_vision_backbone(); del vlm

        tasks_nclasses = {t: get_num_classes_for_task(t) for t in self.tasks}
        probe = MultiTaskProbe(
            backbone=backbone,
            tasks=tasks_nclasses,
            freeze_backbone=freeze_backbone_flag,
            dropout_p=float(mcfg.get("dropout_p", 0.3)),
            deeper_heads=bool(mcfg.get("deeper_head", False)),
            hidden_dim=int(mcfg.get("hidden_dim", 512)),
        )

        # Se la backbone è congelata e k>0, sblocca solo gli ultimi k layer
        if freeze_backbone_flag and unfreeze_k > 0:
            probe.unfreeze_last_backbone_k_layers(
                k=unfreeze_k,
                parts=unfreeze_parts,
                include_embeddings=include_embeddings,
            )
        return probe

    def post_build(self):
        # Istanzia RunningMeans (dopo che self.tasks è definito)
        if self.use_running_means:
            self.rm = RunningMeans(self.tasks, alpha=self.rm_alpha)

        tcfg = self.cfg["train"]
        head_lr = float(tcfg.get("lr", 1e-4))
        backbone_lr = float(tcfg.get("backbone_lr", head_lr))
        weight_decay = float(tcfg.get("weight_decay", 1e-4))

        # Separa parametri backbone vs heads
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters()
                    if p.requires_grad and not n.startswith("backbone.")]
        
        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})

        # Costruisci optimizer + scheduler
        self.optimizer = torch.optim.AdamW(groups, lr=head_lr, weight_decay=weight_decay)
        self.scheduler = None # Lo istanzia il base trainer

    def build_dataloaders(self):
        dcfg = self.cfg["data"]
        base_path   = dcfg.get("base_path", None)
        batch_size  = int(dcfg.get("batch_size", 64))
        num_workers = int(dcfg.get("num_workers", 8))
        use_augmentation = bool(dcfg.get("use_augmentation", True))
        use_sampler = bool(dcfg.get("use_sampler", True))
        tasks_nclasses = {t: get_num_classes_for_task(t) for t in self.tasks}

        # --- DATA AUGMENTATION ---
        train_transforms = None
        if use_augmentation:
            train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            ])
        val_transforms = None

        # --- Dataset TRAIN: sempre bilanciato su emotion=0.33 ---
        desired = {"emotion": 0.33}
        train_dataset, agg_counts = DatasetFactory.create_balanced_multi_task_dataset(
            tasks=self.tasks, split="train",
            base_path=base_path, transform=train_transforms,
            num_classes=tasks_nclasses,
            desired_fractions=desired
        )

        # --- Dataset VAL: multi puro (dedup) ---
        val_dataset, _ = DatasetFactory.create_multi_task_dataset(
            tasks=self.tasks, split="val",
            base_path=base_path, transform=val_transforms,
            num_classes=tasks_nclasses
        )

        # --- Pesi per classe (da counts aggregati del TRAIN BASE, prima della duplicazione) ---
        self.class_weights = {}
        for t in self.tasks:
            counts = agg_counts.get(t) if isinstance(agg_counts, dict) else None
            if counts is None:
                w = np.ones(get_num_classes_for_task(t), dtype=np.float64)
            else:
                w = counts_to_weights(np.asarray(counts, dtype=np.float64))
            self.class_weights[t] = torch.tensor(w, dtype=torch.float32, device=self.device)
        print(f"Pesi classi:{self.class_weights}")
        
        # --- Loss per task (dipende da use_sampler) ---
        if use_sampler:
            # CE *senza* pesi di classe (il bilanciamento è nel sampler)
            self.criterions = {t: nn.CrossEntropyLoss(weight=None).to(self.device) for t in self.tasks}

            # sampler pesato per *campione* usando i pesi di classe per task
            # passa solo i task che hai effettivamente (presenti in self.tasks)
            task_class_weights = {t: self.class_weights.get(t, None) for t in self.tasks}
            train_sampler, sample_weights = build_weighted_sampler(
                dataset=train_dataset,
                task_class_weights=task_class_weights,
                combine="mean",
                min_weight=1e-4,
                normalize=True,
                replacement=True,
            )
        else:
            # CE *con* pesi di classe
            self.criterions = {t: nn.CrossEntropyLoss(weight=self.class_weights[t]).to(self.device) for t in self.tasks}
            train_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False if train_sampler is not None else True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_keep_pil,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_keep_pil,
            persistent_workers=(num_workers > 0),
        )
        return train_loader, val_loader

    def compute_losses(self, batch, train: bool = True) -> dict:
        images_list, targets_list = batch
        y = targets_to_tensors(targets_list, self.tasks, device=self.device)
        out = self.model(images_list)
        logits = out["logits"]  # dict: task -> [B, C]

        losses = {}
        for t in self.tasks:
            target = y[t]                 # [B] con -1 per missing
            logit  = logits[t]            # [B, C]
            mask   = (target != -1)
            if mask.any():
                # calcolo su soli esempi validi (masked loss)
                loss_t = self.criterions[t](logit[mask], target[mask])
            else:
                # nessun valido ⇒ loss 0 ma con gradiente definito
                loss_t = torch.zeros((), device=self.device, requires_grad=True)
            losses[t] = loss_t
        return losses

    # ---------- RunningMeans: pesi dinamici per epoca ----------
    def _compute_task_weights(self) -> Dict[str, float]:
        """Calcola i pesi per task: inverso della EMA, normalizzato per la media.
        Se EMA non inizializzata, usa l'inverso del peso statico.
        """
        if not self.use_running_means or self.rm is None:
            return dict(self.static_task_weights)

        raw = []
        for idx, t in enumerate(self.tasks):
            m = self.rm.get_by_index(idx)
            if m is None:
                raw.append(1.0 / max(self.static_task_weights.get(t, 1.0), 1e-8))
            else:
                raw.append(1.0 / max(float(m), 1e-8))

        avg = sum(raw) / max(1, len(raw))
        return {t: raw[i] / avg for i, t in enumerate(self.tasks)}

    def reduce_losses(self, loss_dict: dict) -> torch.Tensor:
        # Se RunningMeans attivo usa i pesi correnti, altrimenti pesi statici
        use_rm = bool(self.use_running_means and (self.rm is not None))
        weights = self.current_task_weights if use_rm else self.static_task_weights

        total = None
        for t, l in loss_dict.items():
            if torch.isfinite(l):
                w = float(weights.get(t, 1.0))
                total = (w * l) if total is None else total + (w * l)
        if total is None:
            return torch.zeros((), device=self.device, requires_grad=True)
        return total

    def on_train_epoch_start(self, epoch: int, epochs: int):
        # Calcola i pesi per task a inizio epoca (RM o statici)
        self.current_task_weights = self._compute_task_weights()
        print(f"[Weights][Epoch {epoch+1}] " + " | ".join(
            f"{k}={v:.3f}" for k, v in self.current_task_weights.items()
        ))

    def after_compute_losses(self, loss_dict: dict, batch):
        # Aggiorna la EMA per-task SOLO se RM attivo e ci sono esempi validi
        if not (self.use_running_means and (self.rm is not None)):
            return
        try:
            targets_list = batch[1]
        except Exception:
            return
        for idx, t in enumerate(self.tasks):
            try:
                ys = [ti.get(t, -1) for ti in targets_list]
                n_valid = int(sum(1 for y in ys if y is not None and int(y) != -1))
            except Exception:
                n_valid = 0
            if n_valid > 0 and torch.isfinite(loss_dict[t]):
                self.rm.update_by_idx(float(loss_dict[t].detach().item()), idx)

        # ----- extra state (RunningMeans) -----
    def extra_state_dicts(self) -> dict:
        blob = {}
        if getattr(self, "rm", None) is not None:
            blob["running_means"] = {
                "alpha": self.rm.alpha,
                "values": self.rm.values,
                "history": self.rm.history,
                "tasks": self.tasks,
            }
        return blob

    def load_extra_state_dicts(self, blob: dict):
        rm_blob = blob.get("running_means")
        if getattr(self, "rm", None) is not None and rm_blob:
            self.rm.alpha = float(rm_blob.get("alpha", self.rm.alpha))
            self.rm.values = dict(rm_blob.get("values", self.rm.values))
            self.rm.history = dict(rm_blob.get("history", self.rm.history))

    def run_meta(self) -> dict:
        meta = super().run_meta()
        mcfg = self.cfg["model"]
        bb_cfg = (mcfg.get("backbone") or {})
        meta.update({
            "trainer": "multi_task",
            "tasks": self.tasks,
            "running_means": bool(self.rm is not None),
            "backbone": {
                "freeze": bool(bb_cfg.get("freeze", mcfg.get("freeze_backbone", True))),
                "unfreeze_last_k": int(bb_cfg.get("unfreeze_last_k", 0)),
                "unfreeze_parts": str(bb_cfg.get("unfreeze_parts", "all")),
                "include_embeddings": bool(bb_cfg.get("include_embeddings", True)),
            },
        })
        return meta