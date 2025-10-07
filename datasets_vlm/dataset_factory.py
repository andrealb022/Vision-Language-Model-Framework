import os
from pathlib import Path
from typing import Iterable, List, Dict, Type, Optional
from torch.utils.data import ConcatDataset
import numpy as np
import yaml
from .mivia_par_dataset import MiviaParDataset
from .face_dataset import FaceDataset
from .multitask_dataset import MultiTaskDataset, BalancedMultiTaskDataset

# ------------------------- Counts utils -------------------------
def aggregate_counts_from_datasets(
    ds,
    task: str,
    num_classes: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Somma i conteggi di classe per un task su tutti i sotto-dataset di `ds`.
    Nessun default: se non trova nulla → None. Se num_classes è dato, pad/tronca.
    """
    agg: Optional[np.ndarray] = None

    def add_counts(one_ds):
        nonlocal agg
        if not hasattr(one_ds, "get_train_class_counts"):
            return
        raw = one_ds.get_train_class_counts(task)
        if raw is None:
            return
        arr = np.asarray(raw, dtype=np.int64)
        if arr.ndim != 1:
            return

        if agg is None:
            agg = np.zeros_like(arr, dtype=np.int64)

        if arr.size > agg.size:
            tmp = np.zeros(arr.size, dtype=np.int64)
            tmp[:agg.size] = agg
            agg = tmp
        elif arr.size < agg.size:
            tmp = np.zeros(agg.size, dtype=np.int64)
            tmp[:arr.size] = arr
            arr = tmp

        agg += arr

    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            add_counts(sub)
    else:
        add_counts(ds)

    if agg is None:
        return None

    if isinstance(num_classes, int) and num_classes > 0:
        if agg.size < num_classes:
            tmp = np.zeros(num_classes, dtype=np.int64)
            tmp[:agg.size] = agg
            agg = tmp
        elif agg.size > num_classes:
            agg = agg[:num_classes]

    return None if int(agg.sum()) == 0 else agg


# ------------------------- Factory -------------------------
class DatasetFactory:
    """
    Factory per istanziare dataset concreti dai nomi simbolici e per creare
    ConcatDataset in base alla mappa task->datasets caricata **solo** da YAML.
    Nessun default in codice.
    """
    # Mappa caricata da YAML (obbligatoria)
    _task_datasets: Optional[Dict[str, Dict[str, List[str]]]] = None

    # Alias retro-compat (riempiti dal YAML)
    TASK_TO_DATASETS_TRAIN: Dict[str, List[str]] = {}
    TASK_TO_DATASETS_VAL:   Dict[str, List[str]] = {}
    TASK_TO_DATASETS_TEST:  Dict[str, List[str]] = {}

    # Classi note da registrare (estendibile)
    _dataset_registry: Dict[str, Type] = {}
    _registered_dataset_classes = [MiviaParDataset, FaceDataset]

    # ---------------- Registration ----------------
    @classmethod
    def register_dataset_class(cls, dataset_cls: Type) -> None:
        if not hasattr(dataset_cls, "get_available_datasets"):
            raise ValueError(f"{dataset_cls.__name__} non espone get_available_datasets()")
        for name in dataset_cls.get_available_datasets():
            if name in cls._dataset_registry:
                prev = cls._dataset_registry[name]
                raise ValueError(
                    f"Dataset '{name}' già registrato da {prev.__name__}. "
                    f"Tentativo di doppia registrazione da {dataset_cls.__name__}."
                )
            cls._dataset_registry[name] = dataset_cls

    # ---------------- YAML loader (obbligatorio) ----------------
    @classmethod
    def _yaml_path(cls) -> Path:
        """
        Path fisso richiesto: configs/task_datasets.yaml
        (relativo alla root del progetto). Se PYTHONPATH è settato,
        usalo come root di progetto.
        """
        project_root = os.getenv("PYTHONPATH") or "."
        return Path(project_root) / "configs" / "task_datasets.yaml"

    @classmethod
    def load_task_map(cls, *, force: bool = False) -> None:
        """
        Carica la mappa task->datasets da configs/task_datasets.yaml.
        Nessun fallback: se il file non esiste o non è valido, solleva errore.
        Richiede che gli split usati (train/val/test) siano presenti.
        """
        if cls._task_datasets is not None and not force:
            return

        path = cls._yaml_path()
        if not path.exists():
            raise FileNotFoundError(
                f"File YAML per task/datasets non trovato: {path}. "
                f"Crea configs/task_datasets.yaml."
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Formato YAML non valido in {path}: atteso dict radice.")

        # Normalizza e valida
        task_datasets: Dict[str, Dict[str, List[str]]] = {}
        for split, mapping in data.items():
            if split not in ("train", "val", "test"):
                raise ValueError(f"Split non valido '{split}' in {path}. Ammessi: train, val, test.")
            if not isinstance(mapping, dict):
                raise ValueError(f"Sezione '{split}' deve essere una mappa task -> [datasets].")
            task_map_norm: Dict[str, List[str]] = {}
            for task, lst in mapping.items():
                if not isinstance(lst, list) or not all(isinstance(x, str) for x in lst):
                    raise ValueError(f"tasks['{split}']['{task}'] deve essere una lista di stringhe.")
                # deduplica preservando ordine
                seen = set(); ordered = []
                for name in lst:
                    if name not in seen:
                        seen.add(name); ordered.append(name)
                task_map_norm[str(task).lower()] = ordered
            task_datasets[split] = task_map_norm

        # Richiedi almeno train e test se verranno usati (nessun default)
        cls._task_datasets = task_datasets

        # Alias retro-compat (se mancano, lasciali vuoti: nessun default)
        cls.TASK_TO_DATASETS_TRAIN = task_datasets.get("train", {})
        cls.TASK_TO_DATASETS_VAL   = task_datasets.get("val",   {})
        cls.TASK_TO_DATASETS_TEST  = task_datasets.get("test",  {})

    # ---------------- Helpers interni ----------------
    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._task_datasets is None:
            cls.load_task_map()

    @classmethod
    def _task_map_for_split(cls, split: str) -> Dict[str, List[str]]:
        cls._ensure_loaded()
        s = split.lower().strip()
        if s not in cls._task_datasets:
            raise ValueError(
                f"Split '{split}' non definito in configs/task_datasets.yaml. "
                f"Aggiungilo esplicitamente (nessun default)."
            )
        return cls._task_datasets[s]

    # ----------------------------- API pubblica -----------------------------
    @staticmethod
    def get_available_datasets() -> List[str]:
        """Nomi dei dataset registrati nella factory."""
        return list(DatasetFactory._dataset_registry.keys())

    @staticmethod
    def create_dataset(
        dataset_name: str,
        split: str = "train",
        base_path=None,
        transform=None,
        **kwargs,
    ):
        """Istanzia un dataset concreto per lo split dato."""
        if dataset_name not in DatasetFactory._dataset_registry:
            available = DatasetFactory.get_available_datasets()
            raise ValueError(
                f"Dataset '{dataset_name}' non registrato. Disponibili: {sorted(available)}"
            )
        dataset_class = DatasetFactory._dataset_registry[dataset_name]
        return dataset_class(
            dataset_name=dataset_name,
            split=split,
            base_path=base_path,
            transform=transform,
            **kwargs,
        )
        
# ---------------- MultiTask & Balanced creators ----------------
    @staticmethod
    def create_multi_task_dataset(
        tasks: Iterable[str],
        split: str = "train",
        base_path=None,
        transform=None,
        num_classes: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> tuple[MultiTaskDataset, Dict[str, Optional[np.ndarray]]]:
        """
        Ritorna un MultiTaskDataset che unisce i dataset richiesti UNA SOLA VOLTA
        (dedup tra task) e i counts aggregati per task.
        """
        factory = DatasetFactory
        tasks = [t.lower().strip() for t in tasks]
        task_map = factory._task_map_for_split(split)

        unknown = sorted(set(tasks) - set(task_map.keys()))
        if unknown:
            raise ValueError(
                f"Task non supportati per lo split '{split}': {unknown}. "
                f"Definiscili in configs/task_datasets.yaml."
            )

        # dedup dei dataset mantenendo l'ordine
        seen, selected_names = set(), []
        for t in tasks:
            for name in task_map[t]:
                if name not in seen:
                    seen.add(name)
                    selected_names.append(name)
        if not selected_names:
            raise ValueError(f"Nessun dataset selezionato per tasks={tasks} nello split '{split}'")

        # istanzia i sotto-dataset
        instantiated = []
        for name in selected_names:
            if name not in factory._dataset_registry:
                available = factory.get_available_datasets()
                raise ValueError(
                    f"Il dataset '{name}' non è registrato nella factory. "
                    f"Disponibili: {sorted(available)}"
                )
            ds = factory.create_dataset(
                dataset_name=name,
                split=split,
                base_path=base_path,
                transform=transform,
                **kwargs,
            )
            instantiated.append(ds)

        mtd = MultiTaskDataset(instantiated, tasks=tasks)

        # Aggrega i counts per task (con pad/troncamento opzionale)
        num_classes = num_classes or {}
        counts_per_task: Dict[str, Optional[np.ndarray]] = {}
        for t in tasks:
            k = num_classes.get(t)
            counts_per_task[t] = aggregate_counts_from_datasets(mtd, t, num_classes=k)

        return mtd, counts_per_task

    @staticmethod
    def create_balanced_multi_task_dataset(
        tasks: Iterable[str],
        split: str = "train",
        *,
        desired_fractions: Dict[str, float],
        base_path=None,
        transform=None,
        num_classes: Optional[Dict[str, int]] = None,
        duplicate_transform: Optional[callable] = None,
        random_seed: Optional[int] = 0,
        **kwargs,
    ) -> tuple[BalancedMultiTaskDataset, Dict[str, Optional[np.ndarray]]]:
        """
        Crea un MultiTaskDataset deduplicato e lo avvolge con un BalancedMultiTaskDataset
        che duplica campioni con label valida per raggiungere le frazioni desiderate.
        I counts ritornati sono quelli del dataset base (pre-duplicazione).
        """
        factory = DatasetFactory
        mtd, counts = factory.create_multi_task_dataset(
            tasks=tasks,
            split=split,
            base_path=base_path,
            transform=transform,
            num_classes=num_classes,
            **kwargs,
        )

        btd = BalancedMultiTaskDataset(
            base_dataset=mtd,
            tasks=[t.lower().strip() for t in tasks],
            desired_fractions={k.lower().strip(): float(v) for k, v in desired_fractions.items()},
            duplicate_transform=duplicate_transform,
            random_seed=random_seed,
        )
        return btd, counts

for _cls in DatasetFactory._registered_dataset_classes:
    DatasetFactory.register_dataset_class(_cls)
DatasetFactory._ensure_loaded()