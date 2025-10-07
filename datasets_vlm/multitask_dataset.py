from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import random
import numpy as np
from torch.utils.data import ConcatDataset, Dataset

# Costante unica per label mancanti (coerente con i tuoi BaseDataset/FaceDataset)
MISSING_LABEL = -1

# -----------------------------------------------------------------------------
# Helpers: estrazione label da sample "grezzo" senza aprire immagini
# -----------------------------------------------------------------------------

def _labels_from_raw_sample(sample: Any) -> Optional[Dict[str, Any]]:
    """Ritorna il dict delle label da un sample *grezzo* (senza aprire immagini).
    Accetta:
      - dict con chiave 'labels'
      - tuple/list (image, labels) → prende labels
    Altrimenti None.
    """
    if isinstance(sample, dict) and "labels" in sample:
        return sample["labels"]
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[1]
    return None


def _extract_label(labels: Any, task: str) -> int:
    """Estrae la label intera per `task` da `labels`.
    - Se float (es. età in regressione), la consideriamo valida se >= 0 e la
      convertiamo a int troncando (solo per contare/filtrare).
    - Se mancante → MISSING_LABEL.
    """
    missing = MISSING_LABEL
    if isinstance(labels, dict):
        v = labels.get(task, missing)
    else:
        # Supporto minimale ad array/tuple nell'ordine canonico
        order = ["gender", "age", "ethnicity", "emotion"]
        if isinstance(labels, (list, tuple)) and task in order:
            idx = order.index(task)
            v = labels[idx] if idx < len(labels) else missing
        else:
            v = missing

    try:
        if isinstance(v, float):
            return missing if v < 0 else int(v)
        return int(v)
    except Exception:
        return missing


# =============================================================================
# MultiTaskDataset: Concat di più BaseDataset con utility per tasks
# =============================================================================

class MultiTaskDataset(ConcatDataset):
    """
    ConcatDataset con utilità per task multipli:
    - estrazione rapida delle label per task senza caricare immagini, leggendo
      `ds.samples` se disponibile (compatibile con i tuoi dataset);
    - aggregazione dei conteggi per classe dai singoli dataset (via
      `get_train_class_counts`).

    NOTA: la dedup dei dataset condivisi tra task è gestita dalla Factory.
    """

    def __init__(self, datasets: List[Dataset], *, tasks: Iterable[str]) -> None:
        super().__init__(datasets)
        self.tasks: List[str] = [t.lower().strip() for t in tasks]
        self.dataset_names: List[str] = [getattr(d, "name", type(d).__name__) for d in datasets]
        # cache: task -> np.ndarray (len == len(self))
        self._labels_cache: Dict[str, np.ndarray] = {}

    # ------------------------------- Utility ----------------------------------
    def get_all_labels(self, task: str) -> np.ndarray:
        t = task.lower().strip()
        if t in self._labels_cache:
            return self._labels_cache[t]

        arrays: List[np.ndarray] = []
        for ds in self.datasets:
            if hasattr(ds, "samples"):
                raw_list = getattr(ds, "samples")  # type: ignore[attr-defined]
                labels = np.fromiter(
                    (
                        _extract_label(_labels_from_raw_sample(s) or {}, t)
                        for s in raw_list
                    ),
                    dtype=np.int64,
                    count=len(raw_list),
                )
                arrays.append(labels)
            else:
                # fallback (più lento): usa __getitem__
                arr = np.full(len(ds), MISSING_LABEL, dtype=np.int64)
                for i in range(len(ds)):
                    sample = ds[i]
                    lbls = _labels_from_raw_sample(sample) or {}
                    arr[i] = _extract_label(lbls, t)
                arrays.append(arr)

        out = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int64)
        self._labels_cache[t] = out
        return out

    def get_train_class_counts(self, task: str) -> Optional[np.ndarray]:
        """Somma i counts per classe leggendoli dai singoli dataset (se disponibili).
        Usa `BaseDataset.get_train_class_counts`.
        """
        agg: Optional[np.ndarray] = None
        for ds in self.datasets:
            if hasattr(ds, "get_train_class_counts"):
                raw = ds.get_train_class_counts(task)  # type: ignore[attr-defined]
            else:
                raw = None
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.int64).ravel()
            if agg is None:
                agg = np.zeros_like(arr, dtype=np.int64)
            if arr.size > agg.size:
                tmp = np.zeros(arr.size, dtype=np.int64)
                tmp[: agg.size] = agg
                agg = tmp
            elif arr.size < agg.size:
                tmp = np.zeros(agg.size, dtype=np.int64)
                tmp[: arr.size] = arr
                arr = tmp
            agg += arr
        return agg


# =============================================================================
# BalancedMultiTaskDataset: duplica campioni per raggiungere quote per task
# =============================================================================

class BalancedMultiTaskDataset(Dataset):
    """
    Avvolge un dataset di base (tipicamente `MultiTaskDataset`) e *duplica* campioni
    con label valida per specifici task fino a raggiungere una *frazione desiderata*.

    Esempio: desired_fractions={"age":0.7, "gender":0.6} → dopo la duplicazione,
    ~70% dei campioni hanno label età valida e ~60% hanno label gender valida.

    NOTE:
    - Non altera i dati originali; costruisce un indice esteso (con flag duplicato).
    - Puoi passare `duplicate_transform` per applicare augmentation *solo* ai duplicati.
    - Il valore di label mancante è sempre MISSING_LABEL (-1).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        tasks: Iterable[str],
        desired_fractions: Dict[str, float],
        duplicate_transform: Optional[Callable[[Any], Any]] = None,
        random_seed: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.base = base_dataset
        self.tasks = [t.lower().strip() for t in tasks]
        self.desired = {k.lower().strip(): float(v) for k, v in desired_fractions.items()}
        self._dup_tf = duplicate_transform
        if random_seed is not None:
            random.seed(int(random_seed))

        # cache labels per task (senza aprire immagini se possibile)
        self._labels_cache: Dict[str, np.ndarray] = {}
        self._build_labels_cache()

        # indice esteso [(idx_base, is_dup:bool), ...]
        self._index: List[Tuple[int, bool]] = [(i, False) for i in range(len(self.base))]
        self._apply_balancing()

    # ------------------------------ torch API ------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        idx, is_dup = self._index[i]
        sample = self.base[idx]
        if is_dup and self._dup_tf is not None:
            try:
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    x, y = sample[0], sample[1]
                    x_aug = self._dup_tf(x)
                    return (x_aug, y)
                sample_aug = self._dup_tf(sample)
                return sample_aug
            except Exception:
                return sample
        return sample

    # ------------------------------ helpers --------------------------------
    def _build_labels_cache(self) -> None:
        # se il dataset base fornisce get_all_labels, usa quello (MultiTaskDataset)
        for t in self.tasks:
            arr: Optional[np.ndarray] = None
            if hasattr(self.base, "get_all_labels") and callable(getattr(self.base, "get_all_labels")):
                try:
                    arr = getattr(self.base, "get_all_labels")(t)
                    if isinstance(arr, (list, tuple)):
                        arr = np.asarray(arr, dtype=np.int64)
                except Exception:
                    arr = None
            if arr is None:
                # fallback: itera su base[i]
                N = len(self.base)
                arr = np.full(N, MISSING_LABEL, dtype=np.int64)
                for i in range(N):
                    lbls = _labels_from_raw_sample(self.base[i]) or {}
                    arr[i] = _extract_label(lbls, t)
            assert isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == len(self.base)
            self._labels_cache[t] = arr

    def _apply_balancing(self) -> None:
        original_len = len(self._index)
        for t, desired in self.desired.items():
            if not (0.0 < desired < 1.0):
                raise ValueError(f"desired_fractions['{t}'] deve essere in (0,1), got {desired}")
            labels = self._labels_cache.get(t)
            if labels is None:
                raise ValueError(f"Label cache mancante per task '{t}'")

            valid_idx = [i for i, v in enumerate(labels) if int(v) != MISSING_LABEL]
            c = len(valid_idx)
            frac = c / float(original_len) if original_len > 0 else 0.0
            if frac >= desired or original_len == 0:
                continue

            # x = (d*N - c) / (1 - d)
            to_add = int(round((desired * original_len - c) / max(1e-8, 1.0 - desired)))
            if to_add <= 0:
                continue
            chosen = random.choices(valid_idx, k=to_add)
            self._index.extend((j, True) for j in chosen)

        random.shuffle(self._index)