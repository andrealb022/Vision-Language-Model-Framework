from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from probing.train.base_trainer import BaseTrainer
from probing.train.utils import (
    collate_keep_pil,
    get_num_classes_for_task,
    counts_to_weights,
    targets_to_tensors,
)
from probing.train.losses import UncertaintyWeighter
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from probing.models.multitask_probe import MultiTaskProbe


class MultiTaskTrainer(BaseTrainer):
    def __init__(self, cfg: dict, run_name: str, ckpt_root: Path):
        self.tasks = [t.lower() for t in cfg["tasks"]]
        # UW from YAML
        tcfg = cfg["train"]
        uw_cfg = (tcfg.get("uncertainty_weighting") or {})
        self.uw_enabled = bool(uw_cfg.get("enabled", False))
        self.uw_init_logv = float(uw_cfg.get("init_log_var", 0.0))
        self.uw = None
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

        # UW come modulo separato (come prima)
        tcfg = self.cfg["train"]
        uw_cfg = (tcfg.get("uncertainty_weighting") or {})
        self.uw_enabled = bool(uw_cfg.get("enabled", False))
        self.uw_init_logv = float(uw_cfg.get("init_log_var", 0.0))
        self.uw = UncertaintyWeighter(self.tasks, init_log_var=self.uw_init_logv) if self.uw_enabled else None
        return probe

    def post_build(self):
        # Porta UW sul device se abilitata
        if self.uw is not None:
            self.uw.to(self.device)

        tcfg = self.cfg["train"]
        head_lr = float(tcfg.get("lr", 1e-4))
        backbone_lr = float(tcfg.get("backbone_lr", head_lr))
        weight_decay = float(tcfg.get("weight_decay", 1e-4))

        # Separa parametri backbone vs heads (e UW)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = [p for n, p in self.model.named_parameters()
                    if p.requires_grad and not n.startswith("backbone.")]
        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if self.uw is not None:
            groups.append({"params": list(self.uw.parameters()), "lr": head_lr})

        # Costruisci optimizer + scheduler
        self.optimizer = torch.optim.AdamW(groups, lr=head_lr, weight_decay=weight_decay)

        scfg = (tcfg.get("scheduler") or {"type": "cosine_wr", "T_0": 10, "T_mult": 2})
        if (scfg or {}).get("type", "cosine_wr") == "cosine_wr":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=int(scfg.get("T_0", 10)), T_mult=int(scfg.get("T_mult", 2))
            )
        else:
            self.scheduler = None

    def build_dataloaders(self):
        dcfg = self.cfg["data"]
        base_path   = dcfg.get("base_path", None)
        batch_size  = int(dcfg.get("batch_size", 64))
        num_workers = int(dcfg.get("num_workers", 8))
        tasks_nclasses = {t: get_num_classes_for_task(t) for t in self.tasks}

        train_ds, agg_counts = DatasetFactory.create_task_dataset(
            tasks=self.tasks, split="train", base_path=base_path, transform=None, num_classes=tasks_nclasses
        )
        val_ds, _ = DatasetFactory.create_task_dataset(
            tasks=self.tasks, split="val", base_path=base_path, transform=None, num_classes=tasks_nclasses
        )

        # --- Sampler dalla config (TASK-ONLY) ---
        tcfg = self.cfg.get("train", {})
        sampler_cfg = (tcfg.get("sampler") or {})
        use_weighted = (
            str(sampler_cfg.get("type", "")).lower() in {"weighted", "task_only"}
            or bool(sampler_cfg.get("enabled", False))
            or bool(tcfg.get("use_weighted_sampler", False))
        )
        beta = float(sampler_cfg.get("beta", tcfg.get("oversample_beta", 1.0)))
        replacement = bool(sampler_cfg.get("replacement", True))

        sampler = None
        shuffle = True
        if use_weighted:
            per_sample_w = self._build_task_only_sample_weights(train_ds, self.tasks, agg_counts, beta=beta)
            if float(np.sum(per_sample_w)) <= 0.0:
                print("[Sampler] Tutti i pesi = 0 → uso pesi uniformi.")
                per_sample_w = np.ones(len(train_ds), dtype=np.float32)
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(per_sample_w.astype(np.float32)),
                num_samples=len(train_ds),
                replacement=replacement,
            )
            shuffle = False


        # --- Class weights (per task) ---
        self.class_weights = {}
        for t in self.tasks:
            counts = agg_counts.get(t) if isinstance(agg_counts, dict) else None
            if counts is None:
                w = np.ones(get_num_classes_for_task(t), dtype=np.float64)
            else:
                w = counts_to_weights(np.asarray(counts, dtype=np.float64))
            self.class_weights[t] = torch.tensor(w, dtype=torch.float32, device=self.device)

        # --- Loss per task (masked) ---
        self.criterions = {
            t: nn.CrossEntropyLoss(weight=self.class_weights[t], ignore_index=-1).to(self.device)
            for t in self.tasks
        }

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_keep_pil, persistent_workers=(num_workers > 0)
        )
        val_loader   = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            collate_fn=collate_keep_pil, persistent_workers=(num_workers > 0)
        )
        return train_loader, val_loader

    def compute_losses(self, batch, train: bool = True) -> dict:
        images_list, targets_list = batch
        y = targets_to_tensors(targets_list, self.tasks, device=self.device)
        out = self.model(images_list)
        logits = out["logits"]  # dict task -> [B, C]

        losses = {}
        for t in self.tasks:
            loss_t = self.criterions[t](logits[t], y[t])  # ignore_index=-1 nella CE gestisce il masking
            losses[t] = loss_t
        return losses

    def reduce_losses(self, loss_dict: dict) -> torch.Tensor:
        if self.uw is not None:
            return self.uw(loss_dict)
        return super().reduce_losses(loss_dict)

    # ----- extra state (UW) -----
    def extra_state_dicts(self) -> dict:
        return {"uw": self.uw.state_dict()} if self.uw is not None else {}

    def load_extra_state_dicts(self, blob: dict):
        if self.uw is not None and "uw" in blob:
            self.uw.load_state_dict(blob["uw"], strict=False)

    def run_meta(self) -> dict:
        meta = super().run_meta()
        mcfg = self.cfg["model"]
        bb_cfg = (mcfg.get("backbone") or {})
        meta.update({
            "trainer": "multi_task",
            "tasks": self.tasks,
            "uncertainty_weighting": bool(self.uw is not None),
            "sampler_mode": "task_only" if self.cfg.get("train", {}).get("sampler") else "none",
            "backbone": {
                "freeze": bool(bb_cfg.get("freeze", mcfg.get("freeze_backbone", True))),
                "unfreeze_last_k": int(bb_cfg.get("unfreeze_last_k", 0)),
                "unfreeze_parts": str(bb_cfg.get("unfreeze_parts", "all")),
                "include_embeddings": bool(bb_cfg.get("include_embeddings", True)),
            },
        })
        return meta

    # -------------------------------
    # Task-only sample weights helper
    # -------------------------------
    @staticmethod
    def _build_task_only_sample_weights(dataset, tasks: List[str], agg_counts: Dict[str, List[int]], beta: float = 0.99) -> np.ndarray:
        """Costruisce pesi per-sample considerando SOLO la disponibilità del task (ignora le classi).
        - Calcola un peso per task con formula class-balanced: w_t = (1-beta)/(1-beta^{n_t}), dove n_t = #label validi.
        - Peso sample i = media dei w_t dei task per cui il sample ha label != -1. Se nessun task valido, peso=0.
        """
        # 1) Pesi per task basati sui conteggi validi
        w_task: Dict[str, float] = {}
        for t in tasks:
            counts = agg_counts.get(t)
            n_t = int(np.sum(counts)) if counts is not None else 0
            if n_t <= 0:
                w_task[t] = 0.0
            else:
                if beta >= 1.0:
                    # evita divisione per zero: usa inverso della frequenza
                    w_task[t] = 1.0 / float(n_t)
                else:
                    w_task[t] = (1.0 - beta) / (1.0 - (beta ** n_t))
                    
        N = len(dataset)
        weights = np.zeros(N, dtype=np.float32)
        for i in range(N):
            _, ti = dataset[i]
            present_w = []
            for t in tasks:
                yi = ti.get(t, -1)
                if isinstance(yi, torch.Tensor):
                    yi = yi.item()
                if yi is not None and int(yi) != -1:
                    present_w.append(w_task[t])
            if present_w:
                weights[i] = float(np.mean(present_w))
            else:
                weights[i] = 0.0  # nessun task valido
        return weights