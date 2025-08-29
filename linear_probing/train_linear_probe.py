"""
Linear Probe Training Script + LoRA (backbone) + modules_to_save (classifier) + AMP

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
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------
# Utility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_classification(task: str) -> bool:
    return task.lower() in {"gender", "ethnicity", "emotion", "facial emotion"}

def get_num_classes_for_task(task: str) -> int:
    t = task.lower()
    if t == "gender": return 2
    if t in {"emotion", "facial emotion"}: return 7
    if t == "ethnicity": return 4
    raise ValueError(f"Task di classificazione non riconosciuto: {task}")

def stratified_indices(labels, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for idx, y in enumerate(labels): buckets[y].append(idx)
    train_idx, val_idx = [], []
    for _, idxs in buckets.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val]); train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx

# -----------------------
# Collate e conversione target
# -----------------------
def collate_keep_pil(batch):
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list

def targets_to_tensor(targets_list, task: str, device):
    is_cls = is_classification(task)
    ys = []
    if is_cls:
        key = "emotion" if task.lower().startswith("facial") or task.lower() == "emotion" else task
        for tgt in targets_list: ys.append(int(tgt.get(key)))
        y = torch.tensor(ys, dtype=torch.long, device=device)
    else:
        for tgt in targets_list: ys.append(float(tgt.get("age")))
        y = torch.tensor(ys, dtype=torch.float32, device=device).unsqueeze(1)
    return y

# -----------------------
# Argparse
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probe training (PIL → processor) con salvataggi compatti + AMP")
    # Model args
    parser.add_argument("--model_name", type=str, default="llava",
                        choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp32",
                        choices=["4bit", "8bit", "fp16", "fp32"])
    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="VggFace2-Train",
                        choices=DatasetFactory.get_available_datasets())
    parser.add_argument("--task", type=str, default="gender",
                        help="gender | ethnicity | emotion | age")
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # Training args
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA args (backbone); la classifier è salvata densa con modules_to_save
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Se attivo, applica LoRA agli ultimi K layer della backbone; la classifier resta densa e viene salvata con modules_to_save.")
    parser.add_argument("--lora_last_k", type=int, default=2)
    parser.add_argument("--lora_attn_only", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Resume / Output
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Percorso da cui riprendere: directory adapter (LoRA) o directory head (no-LoRA)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Se None: usa <project_root>/linear_probing/checkpoints/")
    return parser.parse_args()

# -----------------------
# Checkpoint helpers
# -----------------------
def save_lora_training_bundle(adapter_dir: Path, probe: nn.Module,
                              optimizer: optim.Optimizer, scheduler, scaler,
                              next_epoch: int, best_val_loss: float, meta: dict, args: argparse.Namespace):
    """
    Salva:
      - adapter LoRA + modules_to_save (classifier) via PEFT
      - training_state.pth con optimizer/scheduler/scaler/epoch/best_val_loss/meta/args
    """
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    probe.save_pretrained(adapter_dir.as_posix())
    bundle = {
        "epoch": next_epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "meta": meta,
        "args": vars(args),
    }
    torch.save(bundle, adapter_dir / "training_state.pth")

def try_resume_from_adapters_dir(resume_path: Path, probe: nn.Module,
                                 optimizer: optim.Optimizer, scheduler, scaler):
    """
    Carica adapter LoRA + classifier (modules_to_save) in 'probe' e, se presente,
    ripristina stato di training. Ritorna (peft_probe, start_epoch, best_val_loss, meta).
    """
    resume_path = Path(resume_path)
    if not resume_path.is_dir():
        raise RuntimeError("resume_from non è una directory di adapter PEFT.")
    # carica PEFT (adapter + modules_to_save)
    peft_probe = PeftModel.from_pretrained(probe, resume_path.as_posix())
    # bundle di training
    bundle_path = resume_path / "training_state.pth"
    start_epoch, best_val_loss, meta = 0, float("inf"), {}
    if bundle_path.exists():
        bundle = torch.load(bundle_path, map_location="cpu")
        if optimizer is not None and bundle.get("optimizer_state") is not None:
            optimizer.load_state_dict(bundle["optimizer_state"])
        if scheduler is not None and bundle.get("scheduler_state") is not None:
            scheduler.load_state_dict(bundle["scheduler_state"])
        if scaler is not None and bundle.get("scaler_state") is not None:
            scaler.load_state_dict(bundle["scaler_state"])
        start_epoch = int(bundle.get("epoch", 0))
        best_val_loss = float(bundle.get("best_val_loss", float("inf")))
        meta = bundle.get("meta", {})
    else:
        print("[RESUME] Nessun training_state.pth nella dir adapter: optimizer/scheduler/scaler nuovi.")
    return peft_probe, start_epoch, best_val_loss, meta

# ---- NO-LoRA: salvataggio COMPATTO solo classifier + training state ----
def save_classifier_training_bundle(head_dir: Path, probe: nn.Module,
                                    optimizer: optim.Optimizer, scheduler, scaler,
                                    next_epoch: int, best_val_loss: float, meta: dict, args: argparse.Namespace):
    """
    Salva:
      - classifier.pt (solo pesi della head densa)
      - training_state.pth con optimizer/scheduler/scaler/epoch/best_val_loss/meta/args
    """
    head_dir = Path(head_dir)
    head_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.classifier.state_dict(), head_dir / "classifier.pt")
    bundle = {
        "epoch": next_epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "meta": meta,
        "args": vars(args),
    }
    torch.save(bundle, head_dir / "training_state.pth")

def try_resume_from_classifier_dir(resume_path: Path, probe: nn.Module,
                                   optimizer: optim.Optimizer, scheduler, scaler):
    """
    Carica classifier.pt nella head di 'probe' e ripristina training_state.pth se presente.
    Ritorna (start_epoch, best_val_loss, meta).
    """
    resume_path = Path(resume_path)
    if not resume_path.is_dir():
        raise RuntimeError("resume_from non è una directory di head (classifier).")
    cls_path = resume_path / "classifier.pt"
    if not cls_path.exists():
        raise RuntimeError("classifier.pt non trovato nella directory di resume.")
    # carica pesi della head
    state = torch.load(cls_path, map_location="cpu")
    probe.classifier.load_state_dict(state, strict=True)

    # bundle
    bundle_path = resume_path / "training_state.pth"
    start_epoch, best_val_loss, meta = 0, float("inf"), {}
    if bundle_path.exists():
        bundle = torch.load(bundle_path, map_location="cpu")
        if optimizer is not None and bundle.get("optimizer_state") is not None:
            optimizer.load_state_dict(bundle["optimizer_state"])
        if scheduler is not None and bundle.get("scheduler_state") is not None:
            scheduler.load_state_dict(bundle["scheduler_state"])
        if scaler is not None and bundle.get("scaler_state") is not None:
            scaler.load_state_dict(bundle["scaler_state"])
        start_epoch = int(bundle.get("epoch", 0))
        best_val_loss = float(bundle.get("best_val_loss", float("inf")))
        meta = bundle.get("meta", {})
    else:
        print("[RESUME] Nessun training_state.pth nella dir head: optimizer/scheduler/scaler nuovi.")
    return start_epoch, best_val_loss, meta

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Output dir
    if args.output_root:
        output_dir = Path(args.output_root)
    elif project_root:
        output_dir = Path(project_root) / "linear_probing" / "checkpoints"
    else:
        output_dir = Path("linear_probing") / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Modello VLM & Backbone ---
    model_id = None
    vlm = VLMModelFactory.create_model(args.model_name, model_id, device, args.quantization)
    backbone = vlm.get_vision_backbone()
    del vlm

    # Task / out dim
    task_lower = args.task.lower()
    is_cls = is_classification(task_lower)
    probe_out = get_num_classes_for_task(task_lower) if is_cls else 1

    # Backbone SEMPRE frozen
    probe = LinearProbe(backbone=backbone, n_out_classes=probe_out, freeze_backbone=True).to(device)

    # --- LoRA (solo backbone) + modules_to_save (classifier densa) ---
    if args.use_lora:
        backbone_strategy = {"last_k": args.lora_last_k, "attn_only": args.lora_attn_only}
        bk_rel = backbone.get_lora_target_names(backbone_strategy)
        bk_full = [f"backbone.{n}" for n in bk_rel]

        # modules_to_save: tutti i Linear della head
        head_dense = [n for n, m in probe.named_modules()
                      if n.startswith("classifier.") and isinstance(m, nn.Linear)]
        if not head_dense:
            head_dense = ["classifier.1"]

        print("LoRA target_modules (backbone):")
        for t in bk_full: print(" -", t)
        print("modules_to_save (classifier):")
        for t in head_dense: print(" -", t)

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=bk_full,
            modules_to_save=head_dense,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        probe = get_peft_model(probe, lora_cfg).to(device)
        try:
            probe.print_trainable_parameters()
        except Exception:
            pass

    # --- Dataset ---
    transform = None
    dataset = DatasetFactory.create_dataset(args.dataset_name, base_path=args.base_path, train=True, transform=transform)

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
        train_idx = train_subset.indices; val_idx = val_subset.indices

    train_ds = Subset(dataset, train_idx); val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False,
                              collate_fn=collate_keep_pil)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False,
                            collate_fn=collate_keep_pil)

    # --- Loss / Optim / Sched / AMP ---
    criterion = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()
    trainable_params = [p for p in probe.parameters() if p.requires_grad]
    print("G:", trainable_params)
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # --- Resume (da directory di adapter o di head) ---
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
            if args.use_lora:
                # resume LoRA: adapter dir
                probe, start_epoch, best_val_loss, meta = try_resume_from_adapters_dir(
                    resume_path, probe, optimizer, scheduler, scaler
                )
                probe = probe.to(device)
                print(f"[RESUME] PEFT dir: {resume_path} | start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}")
            else:
                # resume NO-LoRA: head dir
                start_epoch, best_val_loss, meta = try_resume_from_classifier_dir(
                    resume_path, probe, optimizer, scheduler, scaler
                )
                print(f"[RESUME] HEAD dir: {resume_path} | start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}")
        else:
            print(f"[RESUME] Path non valido (serve directory): {resume_path}. Training da zero.")
    else:
        print("[RESUME] Nessun resume richiesto.")

    # --- Training loop con early stopping (val_loss) ---
    patience_left = args.patience
    history = {"epochs": args.epochs, "task": args.task, "train": [], "val": [], "ckpt": None}

    for epoch in range(start_epoch, args.epochs):
        probe.train()
        train_loss_accum, n_train, train_acc_accum = 0.0, 0, 0.0

        for i, (images_list, targets_list) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Train", unit="batch")
        ):
            targets = targets_to_tensor(targets_list, args.task, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                outputs = probe(images=images_list)
                loss = criterion(outputs, targets)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

            scheduler.step(epoch + i / len(train_loader))

            bs = len(images_list)
            train_loss_accum += loss.detach().item() * bs
            n_train += bs
            if is_cls:
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    train_acc_accum += (preds == targets).float().sum().item()

        train_loss = train_loss_accum / max(1, n_train)
        train_metrics = {"loss": train_loss}
        if is_cls: train_metrics["acc"] = train_acc_accum / max(1, n_train)
        else:      train_metrics["mse"] = train_loss

        # ---- Validation ----
        probe.eval()
        val_loss_accum, n_val, val_acc_accum = 0.0, 0, 0.0
        with torch.no_grad():
            for images_list, targets_list in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} - Val", unit="batch"):
                targets = targets_to_tensor(targets_list, args.task, device)
                with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                    outputs = probe(images=images_list)
                    loss = criterion(outputs, targets)

                bs = len(images_list)
                val_loss_accum += loss.item() * bs
                n_val += bs
                if is_cls:
                    preds = outputs.argmax(dim=1)
                    val_acc_accum += (preds == targets).float().sum().item()

        val_loss = val_loss_accum / max(1, n_val)
        val_metrics = {"loss": val_loss}
        if is_cls: val_metrics["acc"] = val_acc_accum / max(1, n_val)
        else:      val_metrics["mse"] = val_loss

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        print(f"[Epoch {epoch+1}] train: {train_metrics} | val: {val_metrics}")

        # Early stopping su val_loss
        improved = val_loss < best_val_loss - 1e-8
        if improved:
            best_val_loss = val_loss
            patience_left = args.patience
            print(f"[SAVE] Epoch {epoch+1}: miglioramento val_loss → {best_val_loss:.6f}")

            meta = {
                "model_name": args.model_name,
                "quantization": args.quantization,
                "dataset_name": args.dataset_name,
                "task": args.task,
                "use_lora": args.use_lora
            }

            if args.use_lora:
                adapter_dir = (output_dir / f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}_adapters").as_posix()
                save_lora_training_bundle(Path(adapter_dir), probe, optimizer, scheduler, scaler,
                                          next_epoch=epoch+1, best_val_loss=best_val_loss, meta=meta, args=args)
                history["ckpt"] = adapter_dir
                print(f"[SAVE] Adapter PEFT + head (modules_to_save) + training bundle → {adapter_dir}")
            else:
                head_dir = (output_dir / f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}_head").as_posix()
                save_classifier_training_bundle(Path(head_dir), probe, optimizer, scheduler, scaler,
                                                next_epoch=epoch+1, best_val_loss=best_val_loss, meta=meta, args=args)
                history["ckpt"] = head_dir
                print(f"[SAVE] SOLO classifier + training bundle → {head_dir}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] Fermato a epoch {epoch+1}. Best val_loss: {best_val_loss:.6f}")
                break

        # Log progressivo
        with open(output_dir / f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}.train_log.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"Training completato. Best: {history['ckpt']}")
    with open(output_dir / f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}.train_log.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()