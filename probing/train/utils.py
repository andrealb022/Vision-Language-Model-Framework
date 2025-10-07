import os, sys, yaml, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

MISSING_LABEL = -1                                       # <- costante mancante

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

def build_weighted_sampler(
    dataset,
    task_class_weights: Dict[str, torch.Tensor],  # es. {"age": Tensor[K_age], "gender": Tensor[2], "emotion": Tensor[7]}
    *,
    combine: str = "mean",                        # "mean" | "sum" | "max"
    min_weight: float = 1e-4,
    normalize: bool = True,
    replacement: bool = True,
) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    """
    Crea un WeightedRandomSampler pesando *per campione* in base alle classi dei task disponibili.
    - Se il dataset espone get_all_labels(task) lo usa (veloce), altrimenti fallback su __getitem__.
    - Per ogni sample i, raccoglie i pesi dei task con label valida e li combina (mean/sum/max).
    - Se un sample non ha etichette valide per nessun task → min_weight.
    - Pesi normalizzati a media ~1 (opzionale).

    Ritorna: (sampler, weights_cpu)
    """
    tasks: Iterable[str] = list(task_class_weights.keys())
    N = len(dataset)

    # ---- raccogliamo labels per task ----
    labels_per_task: Dict[str, np.ndarray] = {}
    # 1) via get_all_labels se disponibile (veloce)
    for t in tasks:
        arr = None
        if hasattr(dataset, "get_all_labels") and callable(getattr(dataset, "get_all_labels")):
            try:
                arr = getattr(dataset, "get_all_labels")(t)
            except Exception:
                arr = None
        if arr is not None:
            arr = np.asarray(arr, dtype=np.int64).reshape(-1)
            if arr.shape[0] == N:
                labels_per_task[t] = arr

    # 2) fallback: iterazione
    need_fallback = [t for t in tasks if t not in labels_per_task]
    if need_fallback:
        for t in need_fallback:
            labels_per_task[t] = np.full(N, MISSING_LABEL, dtype=np.int64)
        for i in range(N):
            sample = dataset[i]
            # sample atteso (image, labels_dict) o {"labels": {...}}
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                lab = sample[1]
            elif isinstance(sample, dict):
                lab = sample.get("labels", {})
            else:
                lab = {}
            for t in need_fallback:
                try:
                    v = lab.get(t, MISSING_LABEL) if isinstance(lab, dict) else MISSING_LABEL
                    labels_per_task[t][i] = int(v)
                except Exception:
                    labels_per_task[t][i] = MISSING_LABEL

    # ---- pre-elabora i vettori pesi classe per evitare device mismatch ----
    weights_table: Dict[str, Optional[list]] = {}
    for t, w in task_class_weights.items():
        if w is None:
            weights_table[t] = None
        else:
            weights_table[t] = w.detach().cpu().flatten().tolist()

    # ---- calcolo pesi per sample ----
    weights = torch.zeros(N, dtype=torch.float32)
    for i in range(N):
        w_parts = []
        for t in tasks:
            lab_i = int(labels_per_task[t][i])
            table = weights_table.get(t)
            if table is None:
                continue
            if lab_i != MISSING_LABEL and 0 <= lab_i < len(table):
                w_parts.append(float(table[lab_i]))
        if not w_parts:
            w_val = min_weight
        else:
            if combine == "sum":
                w_val = sum(w_parts)
            elif combine == "max":
                w_val = max(w_parts)
            else:  # default mean
                w_val = sum(w_parts) / len(w_parts)
        weights[i] = float(w_val)

    # normalizzazione per stabilità
    if normalize:
        mean_w = float(weights.mean().clamp_min(1e-8))
        weights.div_(mean_w)

    sampler = WeightedRandomSampler(weights=weights.cpu(), num_samples=N, replacement=replacement)
    return sampler, weights.cpu()