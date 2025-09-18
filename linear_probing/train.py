import os, sys
from pathlib import Path
import argparse
import random, numpy as np
import yaml
from dotenv import load_dotenv
load_dotenv()
project_root = os.getenv("PYTHONPATH")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)
import torch
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from linear_probing.linear_probe import LinearProbe

# -----------------------
# Utility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_num_classes_for_task(task: str) -> int:
    t = task.lower()
    if t == "gender": return 2
    if t == "emotion": return 7
    if t == "ethnicity": return 4
    if t == "age": return 9
    raise ValueError(f"Task non riconosciuto: {task}")

def collate_keep_pil(batch):
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list

def targets_to_tensor(targets_list, task: str, device):
    key = "age" if task == "age" else task
    ys = [int(t.get(key)) for t in targets_list]
    return torch.tensor(ys, dtype=torch.long, device=device)

def compute_class_weights(loader, task):
    num_classes = get_num_classes_for_task(task)
    counts = np.zeros(num_classes, dtype=np.int64)
    key = "age" if task == "age" else task
    for _, targets_list in loader:
        ys = [int(t.get(key, -1)) for t in targets_list]
        for y in ys:
            if 0 <= y < num_classes:
                counts[y] += 1
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts.astype(np.float64)
    class_weights = inv * (num_classes / inv.sum())
    return class_weights

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# -----------------------
# Checkpoint helpers (HEAD ONLY)
# -----------------------
def save_classifier_training_bundle(head_dir: Path, probe: nn.Module,
                                    optimizer: optim.Optimizer, scheduler, scaler,
                                    next_epoch: int, best_val_loss: float, meta: dict, cfg_path: str):
    torch.save(probe.classifier.state_dict(), head_dir / "classifier.pt")
    bundle = {
        "epoch": next_epoch,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "meta": meta,
        "config_path": cfg_path,
    }
    torch.save(bundle, head_dir / "training_state.pth")

def try_resume_from_classifier_dir(resume_path: Path, probe: nn.Module,
                                   optimizer: optim.Optimizer, scheduler, scaler):
    resume_path = Path(resume_path)
    if not resume_path.is_dir():
        raise RuntimeError("resume.from non è una directory di head (classifier).")
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
    # args: solo --config
    ap = argparse.ArgumentParser(description="Linear Probe training (YAML config)")
    ap.add_argument("--config", type=str, default="linear_probing/config/train_config.yaml", help="Path al file YAML")
    args = ap.parse_args()

    cfg_path = os.path.join(project_root, args.config) if project_root else args.config
    cfg = load_config(cfg_path)

    # --- MODEL ---
    mcfg         = cfg["model"]
    model_name   = mcfg["name"]
    quantization = mcfg["quantization"]
    freeze_bb    = bool(mcfg.get("freeze_backbone", True))
    dropout_p    = float(mcfg.get("dropout_p", 0.3))
    deeper_head  = bool(mcfg.get("deeper_head", False))
    hidden_dim   = int(mcfg.get("hidden_dim", 512))

    # --- TASK & DATA ---
    task           = cfg["task"].lower()
    dcfg           = cfg["data"]
    base_path      = dcfg.get("base_path", None)
    dataset_mode   = (dcfg.get("dataset_mode", "auto") or "auto").lower()
    dataset_filter = dcfg.get("dataset_filter", []) or []
    transform      = None  # default transform del dataset
    batch_size     = int(dcfg.get("batch_size", 128))
    num_workers    = int(dcfg.get("num_workers", 8))
    val_split      = float(dcfg.get("val_split", 0.2))

    # --- TRAIN ---
    tcfg         = cfg["train"]
    epochs       = int(tcfg.get("epochs", 50))
    lr           = float(tcfg.get("lr", 1e-3))
    weight_decay = float(tcfg.get("weight_decay", 1e-4))
    patience     = int(tcfg.get("patience", 5))
    seed         = int(tcfg.get("seed", 42))
    amp_enabled  = bool(tcfg.get("amp", True))
    scfg         = tcfg.get("scheduler", {"type": "cosine_wr", "T_0": 10, "T_mult": 2})

    # seed & device
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # modello & backbone
    vlm = VLMModelFactory.create_model(model_name, model_id=None, device=device, quantization=quantization)
    backbone = vlm.get_vision_backbone()
    del vlm

    # head / probe
    n_classes = get_num_classes_for_task(task)
    probe = LinearProbe(
        backbone=backbone,
        n_out_classes=n_classes,
        freeze_backbone=freeze_bb,
        dropout_p=dropout_p,
        deeper_head=deeper_head,
        hidden_dim=hidden_dim
    ).to(device)

    # dataset tramite factory (auto o filtro)
    if dataset_mode == "auto":
        dataset = DatasetFactory.create_task_dataset(tasks=[task], base_path=base_path, train=True, transform=transform)
        dataset_name_for_log = "auto"
    elif dataset_mode == "filter":
        if not dataset_filter:
            raise ValueError("dataset_mode='filter' ma 'dataset_filter' è vuoto.")
        ds_list = [DatasetFactory.create_dataset(name, base_path=base_path, train=True, transform=transform)
                   for name in dataset_filter]
        dataset = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)
        dataset_name_for_log = ",".join(dataset_filter)
    else:
        raise ValueError(f"dataset_mode sconosciuto: {dataset_mode}")

    print(f"[INFO] Task: {task} | Datasets: {dataset_name_for_log} | N={len(dataset)}")

    # split
    val_size = max(1, int(val_split * len(dataset)))
    train_size = len(dataset) - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size],
                                            generator=torch.Generator().manual_seed(seed))
    train_ds = Subset(dataset, train_subset.indices)
    val_ds   = Subset(dataset, val_subset.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False,
                              collate_fn=collate_keep_pil)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False,
                            collate_fn=collate_keep_pil)

    # loss, opt, sched
    class_weights = torch.tensor(compute_class_weights(train_loader, task), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    scheduler = None
    if (scfg or {}).get("type", "cosine_wr") == "cosine_wr":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(scfg.get("T_0", 10)), T_mult=int(scfg.get("T_mult", 2)))

    use_amp = amp_enabled and (device.type == "cuda")
    print(f"Use AMP: {use_amp}")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # OUTPUT
    head_type = "deeper" if deeper_head else "linear"
    root = Path(project_root or ".") / "linear_probing" / "checkpoints"
    head_name = f"{model_name}_{quantization}_{dataset_name_for_log}_{task}_{head_type}"
    head_dir = root / head_name
    head_dir.mkdir(parents=True, exist_ok=True)

    # salva copia della config usata
    (head_dir / "head_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # resume
    resume_from = (cfg.get("resume", {}) or {}).get("from", None)
    start_epoch, best_val_loss = 0, float("inf")
    if resume_from:
        from_path = Path(resume_from)
        if from_path.is_dir():
            try:
                start_epoch, best_val_loss, meta = try_resume_from_classifier_dir(from_path, probe, optimizer, scheduler, scaler)
                print(f"[RESUME] {from_path} | start_epoch={start_epoch} | best_val_loss={best_val_loss:.6f}")
            except Exception as e:
                print(f"[RESUME] Fallito: {e}. Procedo da zero.")
        else:
            print(f"[RESUME] Path non valido (serve directory): {from_path}. Training da zero.")
    else:
        print("[RESUME] Nessun resume richiesto.")

    # TRAINING LOOP
    patience_left = patience
    history = {"epochs": epochs, "task": task, "train": [], "val": [], "ckpt": None}

    for epoch in range(start_epoch, epochs):
        probe.train()
        train_loss_accum, n_train, correct = 0.0, 0, 0

        for i, (images_list, targets_list) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Train", unit="batch")):
            y = targets_to_tensor(targets_list, task, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                logits = probe(images=images_list)
                loss = criterion(logits, y)
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            if scheduler:
                scheduler.step(epoch + i / max(1, len(train_loader)))
            bs = len(images_list)
            train_loss_accum += loss.detach().item() * bs
            n_train += bs
            correct += (logits.argmax(1) == y).float().sum().item()

        train_loss = train_loss_accum / max(1, n_train)
        train_acc  = correct / max(1, n_train)

        # validation
        probe.eval()
        val_loss_accum, n_val, val_correct = 0.0, 0, 0
        with torch.no_grad():
            for images_list, targets_list in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Val", unit="batch"):
                y = targets_to_tensor(targets_list, task, device)
                with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                    logits = probe(images=images_list)
                    loss = criterion(logits, y)
                bs = len(images_list)
                val_loss_accum += loss.item() * bs
                n_val += bs
                val_correct += (logits.argmax(1) == y).float().sum().item()

        val_loss = val_loss_accum / max(1, n_val)
        val_acc  = val_correct / max(1, n_val)

        history["train"].append({"loss": float(train_loss), "acc": float(train_acc)})
        history["val"].append({"loss": float(val_loss), "acc": float(val_acc)})
        print(f"[Epoch {epoch+1}] train: loss={train_loss:.4f}, acc={train_acc:.4f} | val: loss={val_loss:.4f}, acc={val_acc:.4f}")

        improved = val_loss < best_val_loss - 1e-8
        if improved:
            best_val_loss = val_loss
            patience_left = patience
            meta = {
                "model_name": model_name,
                "quantization": quantization,
                "dataset_name": dataset_name_for_log,
                "task": task,
                "freeze_backbone": freeze_bb,
                "deeper_head": deeper_head,
                "dropout_p": dropout_p,
                "hidden_dim": hidden_dim,
            }
            save_classifier_training_bundle(head_dir, probe, optimizer, scheduler, scaler,
                                            next_epoch=epoch+1, best_val_loss=best_val_loss, meta=meta, cfg_path=cfg_path)
            history["ckpt"] = str(head_dir)
            print(f"[SAVE] Miglioramento → {head_dir}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] epoch {epoch+1}. Best val_loss: {best_val_loss:.6f}")
                break

        # log progress in YAML
        (head_dir / "train_log.yaml").write_text(yaml.safe_dump(history, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Training completato. Best ckpt: {history['ckpt']}")
    (head_dir / "train_log.yaml").write_text(yaml.safe_dump(history, sort_keys=False, allow_unicode=True), encoding="utf-8")

if __name__ == "__main__":
    main()