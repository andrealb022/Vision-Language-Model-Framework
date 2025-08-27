"""
Linear Probe Training Script

Dataset e task disponibili:
- CelebA_HQ       → gender
- FairFace        → gender, ethnicity
- MiviaGender     → gender
- RAF-DB          → gender, facial emotion
- VggFace2-Train  → gender, age
"""
from dotenv import load_dotenv
import os, sys
load_dotenv()
project_root = os.getenv("PYTHONPATH")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from collections import defaultdict
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from linear_probing.linear_probe import LinearProbe

# -----------------------
# Utility
# -----------------------
def set_seed(seed: int = 42):
    """Imposta il seed per riproducibilità su Python, NumPy e PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_classification(task: str) -> bool:
    """Ritorna True se il task è di classificazione, False se regressione."""
    return task.lower() in {"gender", "ethnicity", "emotion", "facial emotion"}

def get_num_classes_for_task(task: str) -> int:
    """Ritorna il numero di classi attese per un dato task di classificazione."""
    t = task.lower()
    if t == "gender":
        return 2
    if t in {"emotion", "facial emotion"}:
        return 7
    if t == "ethnicity":
        return 4
    raise ValueError(f"Task di classificazione non riconosciuto: {task}")

def stratified_indices(labels, val_ratio=0.2, seed=42):
    """
    Esegue uno split stratificato preservando le proporzioni di classe.

    Args:
        labels (list[int]): etichette di classe per ogni esempio
        val_ratio (float): frazione da destinare alla validation
        seed (int): seme per riproducibilità

    Returns:
        (train_idx, val_idx): liste di indici per train e validation
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for idx, y in enumerate(labels):
        buckets[y].append(idx)
    train_idx, val_idx = [], []
    for _, idxs in buckets.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx

# -----------------------
# Collate e conversione target (PIL → processor in backbone)
# -----------------------

def collate_keep_pil(batch):
    """
    Collate function che mantiene le immagini come PIL.Image.
    Restituisce:
      - images_list: lista di PIL.Image (non stackate)
      - targets_list: lista di dict target originali
    """
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list


def targets_to_tensor(targets_list, task: str, device):
    """
    Converte la lista di dict dei target in un tensore label coerente col task.
    - classificazione: torch.long [B]
    - regressione (age): torch.float32 [B,1]
    """
    is_cls = is_classification(task)
    ys = []
    if is_cls:
        key = "emotion" if task.lower().startswith("facial") or task.lower() == "emotion" else task
        for tgt in targets_list:
            ys.append(int(tgt.get(key)))
        y = torch.tensor(ys, dtype=torch.long, device=device)
    else:
        for tgt in targets_list:
            ys.append(float(tgt.get("age")))
        y = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
    return y

# -----------------------
# Argparse
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probe training (PIL → processor)")
    # Model args
    parser.add_argument("--model_name", type=str, default="llava",
                        choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp32",
                        choices=["4bit", "8bit", "fp16", "fp32"])
    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="VggFace2-Train",
                        choices=DatasetFactory.get_available_datasets())
    parser.add_argument("--task", type=str, default="age",
                        help="gender | ethnicity | emotion | age")
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    # Probe/opt args
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze_after", type=int, default=-1,
                        help="Epoch dopo cui sbloccare la backbone (-1: mai)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Resume (es: "linear_probing/checkpoints/llava_fp32_RAF-DB_gender.pth")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint .pth da cui riprendere il training")

    # Output
    parser.add_argument("--output_root", type=str, default=None,
                        help="Se None: usa <project_root>/linear_probing/checkpoints/")
    return parser.parse_args()


# -----------------------
# Checkpoint helpers
# -----------------------

def save_checkpoint(path: Path, epoch: int, probe: nn.Module,
                    optimizer: optim.Optimizer, scheduler, best_val_loss: float, meta: dict):
    """
    Salva un checkpoint contenente stato del modello, optimizer, scheduler e metadati.
    epoch: prossimo epoch da eseguire alla ripresa (quindi +1 dell'ultimo completato).
    """
    ckpt = {
        "epoch": epoch,
        "model_state": probe.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "meta": meta,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: Path, probe: nn.Module,
                    optimizer: optim.Optimizer, scheduler):
    """
    Carica un checkpoint restituendo (start_epoch, best_val_loss, meta).
    Ripristina stato modello, optimizer e scheduler se presenti.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    probe.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt.get("epoch", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    meta = ckpt.get("meta", {})
    return start_epoch, best_val_loss, meta

# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Output dir + filename
    if args.output_root:
        output_dir = Path(args.output_root)
    elif project_root:
        output_dir = Path(project_root) / "linear_probing" / "checkpoints"
    else:
        output_dir = Path("linear_probing") / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}.pth"
    ckpt_best = output_dir / ckpt_name
    log_path  = output_dir / f"{ckpt_name}.train_log.json"

    # --- Modello VLM & Backbone ---
    model_id = None
    vlm = VLMModelFactory.create_model(args.model_name, model_id, device, args.quantization)
    backbone = vlm.get_vision_backbone()
    del vlm
    # Task e dimensione output
    task_lower = args.task.lower()
    is_cls = is_classification(task_lower)
    if is_cls:
        num_classes = get_num_classes_for_task(task_lower)
        probe_out = num_classes
    else:
        num_classes = None
        probe_out = 1  # regressione (age)

    probe = LinearProbe(backbone=backbone, n_out_classes=probe_out,
                        freeze_backbone=args.freeze_backbone).to(device)

    # --- Dataset ---
    transform = None
    dataset = DatasetFactory.create_dataset(args.dataset_name, base_path=args.base_path,
                                            train=True, transform=transform)

    # Split
    if is_cls:
        labels = []
        for i in range(len(dataset)):
            _, tgt = dataset[i]
            key = "emotion" if task_lower.startswith("facial") or task_lower == "emotion" else args.task
            labels.append(int(tgt[key]))
        train_idx, val_idx = stratified_indices(labels, val_ratio=0.2, seed=args.seed)
    else:
        val_size = max(1, int(0.2 * len(dataset)))
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size],
                                                generator=torch.Generator().manual_seed(args.seed))
        train_idx = train_subset.indices
        val_idx = val_subset.indices

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_keep_pil  # mantiene PIL
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_keep_pil  # mantiene PIL
    )

    # --- Loss & Optim ---
    if is_cls:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()  # MSE per age

    trainable_params = [p for p in probe.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # --- Resume (opzionale) ---
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.is_absolute():
            if project_root:
                resume_path = Path(project_root) / resume_path
        if resume_path.exists():
            start_epoch, best_val_loss, meta = load_checkpoint(resume_path, probe, optimizer, scheduler)
            print(f"[RESUME] Ripreso da: {resume_path} | start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}")
        else:
            print(f"[RESUME] Path checkpoint non trovato: {resume_path}. Training da zero.")
    else:
        print("[RESUME] Nessun checkpoint fornito: training da zero.")

    # --- Training loop con early stopping ---
    patience_left = args.patience
    history = {
        "epochs": args.epochs,
        "task": args.task,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_after": args.unfreeze_after,
        "train": [],
        "val": [],
        "ckpt": str(ckpt_best)
    }

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze backbone opzionale
        if args.unfreeze_after >= 0 and epoch > args.unfreeze_after:
            for p in probe.backbone.parameters():
                p.requires_grad = True
            trainable_params = [p for p in probe.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # ---- Train ----
        probe.train()
        train_loss_accum, n_train, train_acc_accum = 0.0, 0, 0.0

        for i, (images_list, targets_list) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train", unit="batch")
        ):
            targets = targets_to_tensor(targets_list, args.task, device)
            optimizer.zero_grad(set_to_none=True)

            # Passa direttamente la lista di PIL al probe; il processor farà il resto
            outputs = probe(images_list)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Scheduler per-batch: tempo continuo epoch + i/len(loader)
            scheduler.step(epoch + i / len(train_loader))

            bs = len(images_list)
            train_loss_accum += loss.item() * bs
            n_train += bs
            if is_cls:
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    train_acc_accum += (preds == targets).float().sum().item()

        train_loss = train_loss_accum / max(1, n_train)
        train_metrics = {"loss": train_loss}
        if is_cls:
            train_metrics["acc"] = train_acc_accum / max(1, n_train)
        else:
            train_metrics["mse"] = train_loss

        # ---- Validation ----
        probe.eval()
        val_loss_accum, n_val, val_acc_accum = 0.0, 0, 0.0
        with torch.no_grad():
            for images_list, targets_list in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val", unit="batch"
            ):
                targets = targets_to_tensor(targets_list, args.task, device)
                outputs = probe(images_list)
                loss = criterion(outputs, targets)

                bs = len(images_list)
                val_loss_accum += loss.item() * bs
                n_val += bs
                if is_cls:
                    preds = outputs.argmax(dim=1)
                    val_acc_accum += (preds == targets).float().sum().item()

        val_loss = val_loss_accum / max(1, n_val)
        val_metrics = {"loss": val_loss}
        if is_cls:
            val_metrics["acc"] = val_acc_accum / max(1, n_val)
        else:
            val_metrics["mse"] = val_loss

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        print(f"[Epoch {epoch+1}] train: {train_metrics} | val: {val_metrics}")

        # Early stopping su val_loss (o MSE per age)
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            patience_left = args.patience
            print(f"[SAVE] Epoch {epoch+1}: miglioramento val_loss → {best_val_loss:.6f}. Salvataggio in {ckpt_best.name}")
            meta = {
                "model_name": args.model_name,
                "quantization": args.quantization,
                "dataset_name": args.dataset_name,
                "task": args.task,
            }
            # Salva un solo file (sovrascrive il precedente)
            save_checkpoint(ckpt_best, epoch+1, probe, optimizer, scheduler, best_val_loss, meta)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] Fermato a epoch {epoch+1}. Best val_loss: {best_val_loss:.6f}")
                break

        # Log progressivo in JSON
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)

    print(f"Training completato. Best checkpoint: {ckpt_best}")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()