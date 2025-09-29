import os, sys, numpy as np, yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import matplotlib.pyplot as plt

from probing.train.utils import (
    set_seed, targets_to_tensors, collate_keep_pil,
    save_state, load_state, save_training_state, try_resume_training
)

class BaseTrainer:
    """
    Gestisce: device, AMP/GradScaler, optimizer, scheduler, ckpt, loop generico.
    Le sottoclassi implementano build_model(), build_datasets(), compute_losses().
    """
    def __init__(self, cfg: dict, run_name: str, ckpt_root: Path):
        self.cfg = cfg
        self.run_name = run_name
        self.ckpt_dir = ckpt_root / run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # device & seed
        tcfg = cfg["train"]
        set_seed(int(tcfg.get("seed", 42)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_enabled = bool(tcfg.get("amp", True))
        self.autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.bfloat16

        # model & data
        self.model = self.build_model().to(self.device)
        self.train_loader, self.val_loader = self.build_dataloaders()

        # optimizer & scheduler
        self.optimizer = None
        self.scheduler = None
        self.post_build()

        # optimizer & scheduler (fallback se non impostati nel post_build)
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.AdamW(
                params,
                lr=float(tcfg.get("lr", 1e-4)),
                weight_decay=float(tcfg.get("weight_decay", 1e-4)),
            )
        if self.scheduler is None:
            scfg = tcfg.get("scheduler", {"type":"cosine_wr", "T_0": 10, "T_mult": 2})
            self.scheduler = (
                CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=int(scfg.get("T_0", 10)),
                    T_mult=int(scfg.get("T_mult", 2))
                )
                if (scfg or {}).get("type","cosine_wr")=="cosine_wr" else None
            )

        # AMP scaler (nuova API)
        self.scaler = torch.amp.GradScaler(
            device="cuda", enabled=self.amp_enabled and self.device.type=="cuda"
        )

        # checkpoint files
        self.model_file = self.ckpt_dir / "model.pt"
        self.state_file = self.ckpt_dir / "training_state.pth"

        # salva copia della config nella cartella del run
        (self.ckpt_dir / "head_config.yaml").write_text(
            yaml.safe_dump(self.cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8"
        )
        self.history = {"train": [], "val": []}

    # ----- API da implementare nelle sottoclassi -----
    def build_model(self) -> nn.Module: raise NotImplementedError
    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]: raise NotImplementedError
    def compute_losses(self, batch, train: bool = True) -> dict: raise NotImplementedError
    def post_build(self): pass  # opzionale

    # ----- Hooks per salvare extra state (es. UW) -----
    def extra_state_dicts(self) -> dict: return {}
    def load_extra_state_dicts(self, blob: dict): pass

    # ----- Fit loop -----
    def fit(self):
        epochs = int(self.cfg["train"].get("epochs", 30))
        patience = int(self.cfg["train"].get("patience", 5))
        start_epoch, best_val = 0, float("inf")

        # Resume pesi modello
        blob = load_state(self.model_file)
        if blob is not None:
            if "model" in blob:
                self.model.load_state_dict(blob["model"], strict=False)
            else:
                self.model.load_state_dict(blob, strict=False)
            self.load_extra_state_dicts(blob)
            print(f"[RESUME] Pesi modello caricati da {self.model_file}")
        start_epoch, best_val = try_resume_training(
            self.state_file, self.optimizer, self.scheduler, self.scaler
        )

        patience_left = patience
        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_monitor = self.train_one_epoch(epoch, epochs)
            self.history["train"].append(train_monitor)

            # VALIDATION OGNI N EPOCHE
            do_val = ((epoch + 1) % int(self.cfg["train"].get("eval_every", 2)) == 0)
            if do_val:
                val_monitor = self.validate_epoch(epoch, epochs)
                self.history["val"].append(val_monitor)
            else:
                # metti un NaN per mantenere allineamento con le epoche
                self.history["val"].append(float("nan"))

            # Early stop & save best sul valore di validation SOLO quando valutato
            if do_val:
                improved = val_monitor < best_val - 1e-8
                if improved:
                    best_val = val_monitor
                    patience_left = patience
                    blob = {"model": self.model.state_dict()} | self.extra_state_dicts()
                    save_state(self.model_file, blob)
                    save_training_state(
                        self.state_file, self.optimizer, self.scheduler, self.scaler,
                        next_epoch=epoch+1, best_val=best_val, meta=self.run_meta(), cfg_path=self.cfg_path()
                    )
                    print(f"[SAVE] Miglioramento → {self.model_file} (monitor={val_monitor:.6f})")
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"[EARLY STOP] epoch {epoch+1}. Best monitor: {best_val:.6f}")
                        break
        self._save_history_csv()
        self._save_history_plot()

    def train_one_epoch(self, epoch: int, epochs: int) -> float:
        running = self._init_agg()
        for i, batch in enumerate(tqdm(self.train_loader, desc=f"Train {epoch+1}/{epochs}", unit="batch")):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=("cuda" if self.device.type=="cuda" else "cpu"),
                dtype=self.autocast_dtype, enabled=self.amp_enabled
            ):
                loss_dict = self.compute_losses(batch, train=True)
                loss = self.reduce_losses(loss_dict)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                loss.backward(); self.optimizer.step()

            if self.scheduler:
                # step continuo (per cosine WR)
                self.scheduler.step(epoch + i / max(1, len(self.train_loader)))

            self._accumulate(running, loss_dict, batch)

        monitor = self._epoch_log("train", running)
        return monitor

    @torch.no_grad()
    def validate_epoch(self, epoch: int, epochs: int) -> float:
        self.model.eval()
        running = self._init_agg()
        for batch in tqdm(self.val_loader, desc=f"Val {epoch+1}/{epochs}", unit="batch"):
            loss_dict = self.compute_losses(batch, train=False)
            self._accumulate(running, loss_dict, batch)

        monitor = self._epoch_log("val", running)
        return monitor

    # ----- helpers -----
    def reduce_losses(self, loss_dict: dict) -> torch.Tensor:
        """Default: somma delle loss (sovrascrivibile dalle sottoclassi)."""
        return sum(loss_dict.values())

    def _init_agg(self):
        return {"sum": {}, "n": {}}

    def _accumulate(self, running, loss_dict, batch):
        # Assume batch[1] è targets_list come nel collate_keep_pil (se features mode, le sottoclassi sovrascrivono compute_losses)
        targets_list = None
        if isinstance(batch, (list, tuple)) and len(batch) > 1:
            targets_list = batch[1]

        for k, v in loss_dict.items():
            # conteggia solo esempi validi (y!=-1) se abbiamo targets_list
            n = 1
            if targets_list is not None:
                try:
                    ys = [t.get(k, -1) for t in targets_list]
                    n = int(sum(1 for y in ys if y is not None and int(y) != -1))
                except Exception:
                    n = len(targets_list)
            running["sum"][k] = running["sum"].get(k, 0.0) + float(v.detach().item()) * max(1, n)
            running["n"][k]   = running["n"].get(k, 0) + max(1, n)

    def _epoch_log(self, split: str, running) -> float:
        keys = sorted(running["sum"].keys())
        if not keys:
            print(f"[{split}] Nessuna loss aggregata")
            return float("inf")
        logs = []
        vals = []
        for k in keys:
            mean = running["sum"][k] / max(1, running["n"][k])
            logs.append(f"{k}: {mean:.4f}")
            vals.append(mean)
        print(f"[{split.upper()}] " + " | ".join(logs) + f" | monitor(mean)={np.mean(vals):.6f}")
        return float(np.mean(vals))

    def _save_history_csv(self):
        # Salva history su CSV
        csv_path = self.ckpt_dir / "history.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i, (tr, va) in enumerate(zip(self.history["train"], self.history["val"]), start=1):
                tr_str = f"{tr:.6f}" if np.isfinite(tr) else ""
                va_str = f"{va:.6f}" if np.isfinite(va) else ""
                f.write(f"{i},{tr_str},{va_str}\n")
        print(f"[HISTORY] CSV salvato in: {csv_path}")

    def _save_history_plot(self):
        # Grafico loss train/val
        epochs = np.arange(1, len(self.history["train"]) + 1)
        train_vals = np.array(self.history["train"], dtype=float)
        val_vals   = np.array(self.history["val"], dtype=float)

        plt.figure(figsize=(7.5, 4.5))
        plt.plot(epochs, train_vals, label="train")
        # Per i NaN di val (epoche senza validazione) matplotlib disegna gap
        plt.plot(epochs, val_vals, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(self.run_name)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out = self.ckpt_dir / "loss_curve.png"
        plt.savefig(out)
        plt.close()
        print(f"[HISTORY] Grafico salvato in: {out}")

    def run_meta(self) -> dict:
        mcfg = self.cfg["model"]
        return {
            "model_name": mcfg["name"],
            "quantization": mcfg.get("quantization"),
        }

    def cfg_path(self) -> str:
        return self.cfg.get("_cfg_path", "unknown")