from typing import Iterable, List, Dict, Type
from torch.utils.data import ConcatDataset
from .mivia_par_dataset import MiviaParDataset
from .face_dataset import FaceDataset
import numpy as np

def aggregate_counts_from_datasets(ds, task: str, num_classes: int) -> np.ndarray | None:
    """Somma i count (train/class_counts.json) su tutti i sotto-dataset."""
    agg = np.zeros(num_classes, dtype=np.int64)

    def add_counts(one_ds):
        nonlocal agg
        if hasattr(one_ds, "get_train_class_counts"):
            lst = one_ds.get_train_class_counts(task)
            if lst is None:
                return
            arr = np.array(lst, dtype=np.int64)
            if arr.size < num_classes:
                tmp = np.zeros(num_classes, dtype=np.int64); tmp[:arr.size] = arr; arr = tmp
            elif arr.size > num_classes:
                arr = arr[:num_classes]
            agg += arr

    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            add_counts(sub)
    else:
        add_counts(ds)

    return None if agg.sum() == 0 else agg

class DatasetFactory:
    """
    Factory statica per istanziare dataset (estendono BaseDataset) a partire dal nome.

    - Mantiene un registro nome_dataset -> classe_dataset
    - Espone:
        * create_dataset(name, *, split, base_path, transform, **kwargs)
        * create_task_dataset(tasks, *, split, base_path, transform, **kwargs)
    """

    _dataset_registry: Dict[str, Type] = {}

    # Classi dataset da registrare (estendibile)
    _registered_dataset_classes = [MiviaParDataset, FaceDataset]

    # Task → datasets consigliati (nomi come in get_available_datasets delle classi)
    TASK_TO_DATASETS_TRAIN: Dict[str, List[str]] = {
        "gender":    ["CelebA_HQ", "FairFace", "RAF-DB", "VggFace2-Train"],
        "ethnicity": ["FairFace"],
        "emotion":   ["RAF-DB"],
        "age":       ["VggFace2-Train", "FairFace"],
    }
    TASK_TO_DATASETS_TEST: Dict[str, List[str]] = {
        "gender":    ["CelebA_HQ", "FairFace", "RAF-DB", "VggFace2-Test", "LFW", "UTKFace"],
        "ethnicity": ["FairFace"],
        "emotion":   ["RAF-DB"],
        "age":       ["VggFace2-Test", "FairFace", "UTKFace"],
    }
    # Popola il registro interrogando le classi
    for dataset_cls in _registered_dataset_classes:
        if hasattr(dataset_cls, "get_available_datasets"):
            for name in dataset_cls.get_available_datasets():
                _dataset_registry[name] = dataset_cls

    # ----------------------------- API -----------------------------
    @staticmethod
    def create_dataset(
        dataset_name: str,
        split: str = "train",
        base_path=None,
        transform=None,
        **kwargs,
    ):
        """
        Istanzia un dataset per uno split specifico.

        Args:
            dataset_name: nome simbolico (es. "MiviaPar", "RAF-DB", "FairFace").
            split: 'train' | 'val' | 'test'.
            base_path: radice del dataset su disco.
            transform: trasformazioni immagine.
            **kwargs: argomenti aggiuntivi passati al costruttore della classe (es. age_is_regression).

        Returns:
            BaseDataset (istanza concreta della classe registrata).
        """
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

    @staticmethod
    def get_available_datasets() -> List[str]:
        """Ritorna i nomi dei dataset registrati nella factory."""
        return list(DatasetFactory._dataset_registry.keys())

    @staticmethod
    def create_task_dataset(
        tasks: Iterable[str],
        split: str = "train",
        base_path=None,
        transform=None,
        num_classes: Dict[str, int] = {},
        **kwargs,
    ) -> tuple[ConcatDataset, dict[str, np.ndarray | None]]:
        """
        Crea un ConcatDataset che unisce (senza duplicati) i dataset suggeriti dai task richiesti,
        e ritorna anche i counts aggregati per ciascun task.
        """
        tasks = [t.lower().strip() for t in tasks]
        valid = set(DatasetFactory.TASK_TO_DATASETS_TRAIN.keys())
        unknown = sorted(set(tasks) - valid)
        if unknown:
            raise ValueError(f"Task non supportati: {unknown}. Task validi: {sorted(valid)}")

        selected_names: list[str] = []
        seen = set()
        for t, ds_names in DatasetFactory.TASK_TO_DATASETS_TRAIN.items():
            if t in tasks:
                for name in ds_names:
                    if name not in seen:
                        seen.add(name)
                        selected_names.append(name)

        if not selected_names:
            raise ValueError(f"Nessun dataset selezionato dai task: {tasks}")

        print(f"[Info] Task {tasks} → datasets: {selected_names}")

        instantiated = []
        for name in selected_names:
            if name not in DatasetFactory._dataset_registry:
                raise ValueError(
                    f"Il dataset '{name}' (richiesto da task {tasks}) non è registrato nella factory."
                )
            ds = DatasetFactory.create_dataset(
                name,
                split=split,
                base_path=base_path,
                transform=transform,
                **kwargs,
            )
            instantiated.append(ds)

        concat_ds = ConcatDataset(instantiated)

        # calcola i counts per ogni task richiesto
        counts_per_task = {
            t: aggregate_counts_from_datasets(concat_ds, t, num_classes.get(t, 0)) for t in tasks
        }
        return concat_ds, counts_per_task

# Esempio d'uso
if __name__ == "__main__":
    print("Dataset disponibili:", sorted(DatasetFactory.get_available_datasets()))
    ds = DatasetFactory.create_dataset("RAF-DB", split="train")
    ds_multi = DatasetFactory.create_task_dataset(["gender", "age"], split="train")