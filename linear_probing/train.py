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
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from linear_probing.linear_probe import LinearProbe

# -------------------------------------------------
# Dataset → Task mapping (gender ha anche "VggFace2-Train")
# -------------------------------------------------
DATASETS_BY_TASK = {
    "gender":    ["CelebA_HQ", "FairFace", "MiviaGender", "RAF-DB", "VggFace2-Train"],
    "ethnicity": ["FairFace"],
    "emotion":   ["RAF-DB"],
    "age":       ["VggFace2-Train", "FairFace"],
}

# -----------------------
# Utility
# -----------------------
def build_dataset_for_task(task: str, base_path: str, train: bool, transform=None,
                           dataset_name: str = None):
    """
    Se dataset_name è:
      - "auto" → concatena tutti i dataset compatibili col task.
      - Un nome singolo → usa solo quello.
      - Una lista separata da virgole → concatena quelli elencati (e compatibili col task).
    """
    all_dsets_for_task = DATASETS_BY_TASK.get(task, [])
    if not all_dsets_for_task:
        raise ValueError(f"Nessun dataset configurato per il task: {task}")

    if dataset_name is None:
        dataset_name = "auto"

    dataset_name = dataset_name.strip()
    if dataset_name.lower() == "auto":
        selected = all_dsets_for_task
    elif "," in dataset_name:
        req = [x.strip() for x in dataset_name.split(",")]
        selected = [d for d in req if d in all_dsets_for_task]
        missing = set(req) - set(selected)
        if missing:
            print(f"[WARN] Questi dataset non supportano '{task}': {sorted(missing)} — ignorati.")
        if not selected:
            raise ValueError("Nessun dataset valido dopo il filtro.")
    else:
        if dataset_name not in all_dsets_for_task:
            raise ValueError(
                f"'{dataset_name}' non supporta il task '{task}'. "
                f"Usa uno tra: {all_dsets_for_task} oppure '--dataset_name auto'."
            )
        selected = [dataset_name]

    datasets = [
        DatasetFactory.create_dataset(name, base_path=base_path, train=train, transform=transform)
        for name in selected
    ]
    if len(datasets) == 1:
        return datasets[0], selected
    else:
        return ConcatDataset(datasets), selected

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_num_classes_for_task(task: str) -> int:
    if task == "gender": return 2
    if task == "emotion": return 7
    if task == "ethnicity": return 4
    if task == "age": return 9  # 9 classi d'età
    raise ValueError(f"Task di classificazione non riconosciuto: {task}")

# -----------------------
# Collate, targets e pesi
# -----------------------
def collate_keep_pil(batch):
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list

def targets_to_tensor(targets_list, task: str, device):
    """
    Tutti i task sono classificazione (age incluso).
    Si assume che targets_list contenga per 'age' un indice intero 0..8.
    """
    ys = [int(tgt.get(task if task != "age" else "age")) for tgt in targets_list]
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return y

def compute_class_weights(loader, task):
    num_classes = get_num_classes_for_task(task)
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, targets_list in loader:
        ys = [int(t.get(task, -1)) for t in targets_list]
        for y in ys:
            if 0 <= y < num_classes:
                counts[y] += 1
    counts = np.maximum(counts, 1)                           # evita divisioni per zero
    inv = 1.0 / counts.astype(np.float64)                    # inverse frequency
    class_weights = inv * (num_classes / inv.sum())          # normalize
    return class_weights
# -----------------------
# Argparse
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear Probe training (PIL → processor), backbone frozen, salvataggi compatti HEAD + AMP"
    )
    # Model args
    parser.add_argument("--model_name", type=str, default="llava",
                        choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp32",
                        choices=["4bit", "8bit", "fp16", "fp32"])
    parser.add_argument("--deeper_head", type=bool, default=False)

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="auto", help=("Nome del dataset da usare. Può essere: "
              " - uno dei dataset supportati (es. 'FairFace'), "
              " - 'auto' per concatenare tutti i dataset che hanno il task scelto, "
              " - una lista separata da virgole (es. 'FairFace,RAF-DB').")
    )
    parser.add_argument("--task", type=str, default="emotion",
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

    # Resume / Output (./linear_probing/checkpoints/llava_fp32_auto_age_head)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Percorso da cui riprendere: directory head (solo classifier)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Se None: usa <project_root>/linear_probing/checkpoints/")
    return parser.parse_args()

# -----------------------
# Checkpoint helpers (HEAD ONLY)
# -----------------------
def save_classifier_training_bundle(head_dir: Path, probe: nn.Module,
                                    optimizer: optim.Optimizer, scheduler, scaler,
                                    next_epoch: int, best_val_loss: float, meta: dict, args: argparse.Namespace):
    """
    Salva:
      - classifier.pt (solo pesi della head densa)
      - training_state.pth con optimizer/scheduler/scaler/epoch/best_val_loss/meta/args
    """
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
    state = torch.load(cls_path, map_location="cpu", weights_only=True)
    probe.classifier.load_state_dict(state, strict=True)

    bundle_path = resume_path / "training_state.pth"
    start_epoch, best_val_loss, meta = 0, float("inf"), {}
    if bundle_path.exists():
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
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
    deeper = f"deeper" if args.deeper_head else "linear"
    head_dir = Path((output_dir / f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}_head_{deeper}").as_posix())
    head_dir.mkdir(parents=True, exist_ok=True)

    # --- Modello VLM & Backbone ---
    model_id = None
    vlm = VLMModelFactory.create_model(args.model_name, model_id, device, args.quantization)
    backbone = vlm.get_vision_backbone()
    del vlm

    # Task / out dim
    task_lower = args.task.lower()
    probe_out = get_num_classes_for_task(task_lower)

    # Backbone SEMPRE frozen
    probe = LinearProbe(backbone=backbone, n_out_classes=probe_out, freeze_backbone=True, deeper_head=args.deeper_head).to(device)

    # --- Dataset (auto-merge per task) ---
    transform = None
    dataset, used_dsets = build_dataset_for_task(args.task, base_path=args.base_path, train=True,
                                                 transform=transform, dataset_name=args.dataset_name)
    print(f"[INFO] Dataset totali usati per il task {task_lower}: {used_dsets} | dimensione complessiva: {len(dataset)}")

    # --- Split ---
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

    # --- Calcolo class weights direttamente dal train_loader ---
    class_weights = torch.tensor(compute_class_weights(train_loader, task_lower), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # weighted CE
    trainable_params = [p for p in probe.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    use_amp = (device.type == "cuda")
    print(f"Use amp: {use_amp}")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # --- Resume (HEAD only) ---
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_dir():
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
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                train_acc_accum += (preds == targets).float().sum().item()

        train_loss = train_loss_accum / max(1, n_train)
        train_metrics = {"loss": train_loss, "acc": train_acc_accum / max(1, n_train)}

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
                preds = outputs.argmax(dim=1)
                val_acc_accum += (preds == targets).float().sum().item()

        val_loss = val_loss_accum / max(1, n_val)
        val_metrics = {"loss": val_loss, "acc": val_acc_accum / max(1, n_val)}

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
                "task": args.task
            }

            save_classifier_training_bundle(head_dir, probe, optimizer, scheduler, scaler,
                                            next_epoch=epoch+1, best_val_loss=best_val_loss, meta=meta, args=args)
            history["ckpt"] = str(head_dir)
            print(f"[SAVE] SOLO classifier + training bundle → {head_dir}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] Fermato a epoch {epoch+1}. Best val_loss: {best_val_loss:.6f}")
                break

        # Log progressivo
        with open(head_dir / "train_log.json", "w") as f:
            json.dump(history, f, indent=2, default=str)


    print(f"Training completato. Best: {history['ckpt']}")
    with open(head_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2, default=str)


if __name__ == "__main__":
    main()