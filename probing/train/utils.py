import os, sys, yaml, random, numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------------- Utils & Config ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_num_classes_for_task(task: str) -> int:
    t = task.lower()
    if t == "gender": return 2
    if t == "emotion": return 7
    if t == "ethnicity": return 4
    if t == "age": return 9
    raise ValueError(f"Task non riconosciuto: {task}")

def collate_keep_pil(batch):
    images_list  = [b[0] for b in batch]   # PIL.Image list
    targets_list = [b[1] for b in batch]   # dict per-sample
    return images_list, targets_list

def targets_to_tensors(targets_list: List[dict], tasks: List[str], device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Converte la lista di target-dict in tensori Long per task. Usa -1 per label mancanti."""
    out = {}
    for task in tasks:
        key = "age" if task == "age" else task
        ys = []
        for t in targets_list:
            v = t.get(key, None)
            ys.append(int(v) if v is not None else -1)
        tt = torch.tensor(ys, dtype=torch.long)
        out[task] = tt.to(device) if device is not None else tt
    return out

# ---------------- Class/Sample Weights ----------------
def counts_to_weights(counts: np.ndarray) -> np.ndarray:
    """w_i = (1/max(c_i,1)) * (C / sum_j 1/max(c_j,1))  -> media=1"""
    counts = np.maximum(counts.astype(np.float64), 1.0)
    inv = 1.0 / counts
    C = len(counts)
    return inv * (C / inv.sum())

def build_per_sample_weights(dataset, tasks: List[str], agg_counts, beta: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """
    w_i ∝ Σ_t 1[y_i,t != -1] * (1 / freq_t)^beta
    - 1 sola passata sui metadati del dataset
    - normalizza a media ~1
    """
    tasks = [t.lower() for t in tasks]
    # freq per task (se mancano, metti 1 per evitare div/0)
    freq = {t: float(max(1, int(np.sum(agg_counts.get(t, []) if isinstance(agg_counts, dict) else [])))) for t in tasks}
    inv_pow = {t: (1.0 / freq[t]) ** beta for t in tasks}

    N = len(dataset)
    w = np.zeros(N, dtype=np.float32)
    for i in range(N):
        _, tgt = dataset[i]
        s = 0.0
        for t in tasks:
            key = "age" if t == "age" else t
            v = tgt.get(key, None)
            if v is not None and int(v) != -1:
                s += inv_pow[t]
        if s <= 0.0:
            s = min(inv_pow.values()) if len(inv_pow) > 0 else 1.0
        w[i] = s

    mean_w = float(np.mean(w)) + eps
    w /= mean_w
    return w

# ---------------- Checkpoint helpers ----------------
def save_state(model_path: Path, state_dicts: dict):
    """Salva un blob con più state_dict (es: {'model':..., 'uw':...})."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dicts, model_path)

def load_state(model_path: Path) -> dict | None:
    if not model_path.exists():
        return None
    return torch.load(model_path, map_location="cpu")

def save_training_state(state_path: Path, optimizer, scheduler, scaler,
                        next_epoch: int, best_val: float, meta: dict, cfg_path: str):
    blob = {
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": next_epoch,
        "best_val": best_val,
        "meta": meta,
        "config_path": str(cfg_path),
    }
    torch.save(blob, state_path)

def try_resume_training(state_path: Path, optimizer, scheduler, scaler) -> tuple[int, float]:
    start_epoch, best_val = 0, float("inf")
    if not state_path.exists():
        return start_epoch, best_val
    st = torch.load(state_path, map_location="cpu")
    if optimizer is not None and st.get("optimizer_state") is not None:
        optimizer.load_state_dict(st["optimizer_state"])
    if scheduler is not None and st.get("scheduler_state") is not None:
        scheduler.load_state_dict(st["scheduler_state"])
    if scaler is not None and st.get("scaler_state") is not None:
        scaler.load_state_dict(st["scaler_state"])
    start_epoch = int(st.get("epoch", 0))
    best_val    = float(st.get("best_val", float("inf")))
    print(f"[RESUME] Stato training da {state_path} | start_epoch={start_epoch} | best_val={best_val:.6f}")
    return start_epoch, best_val
