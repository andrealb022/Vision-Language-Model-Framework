from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Callable, Any, Dict, List, Optional
import json

# Percorso base di default per dataset organizzati in cartelle standard
BASE_PATH = Path("~/datasets_with_standard_labels/").expanduser()
IMAGES_DIR = "images"
LABELS_FILE = "labels.csv"

class BaseDataset(Dataset, ABC):
    """
    Dataset base astratto per immagini + etichette su disco (compatibile con DataLoader).

    Struttura attesa:
        base_path/
        └── dataset_name/
            ├── train/
            │   ├── images/
            │   └── labels.csv
            ├── val/
            │   ├── images/
            │   └── labels.csv
            └── test/
                ├── images/
                └── labels.csv

    Le sottoclassi DEVONO implementare:
      - _load_labels(): lista di dict {"image_path": Path, "labels": ...}
      - get_labels_from_text_output(output): normalizza l’output del VLM alle stesse etichette
    """

    def __init__(self,
        dataset_name: str,
        split: str = "train",                       # 'train' | 'val' | 'test'
        base_path: Optional[Path] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            dataset_name: nome della sottocartella del dataset.
            split: 'train' | 'val' | 'test'.
            base_path: radice che contiene i dataset; default: BASE_PATH.
            transform: trasformazioni da applicare all'immagine.
        """
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split non valido: {split!r}. Ammessi: 'train'|'val'|'test'.")

        self.name: str = dataset_name
        self.split: str = split
        self.transform = transform
        self.base_path = Path(base_path).expanduser() if base_path else BASE_PATH

        # Percorsi
        self.dataset_path = self.base_path / self.name / self.split
        self.image_folder = self.dataset_path / IMAGES_DIR
        self.label_file = self.dataset_path / LABELS_FILE

        # Controlli essenziali
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"[{self.__class__.__name__}] split '{self.split}' non trovato: {self.dataset_path}")
        if not self.image_folder.exists():
            raise FileNotFoundError(f"[{self.__class__.__name__}] cartella immagini mancante: {self.image_folder}")
        if not self.label_file.exists():
            raise FileNotFoundError(f"[{self.__class__.__name__}] file etichette mancante: {self.label_file}")

        # Carica campioni (lista di dict)
        self.samples: List[Dict[str, Any]] = self._load_labels()
        if not isinstance(self.samples, list):
            raise TypeError(f"[{self.__class__.__name__}] _load_labels() deve restituire list[dict], ottenuto: {type(self.samples)}")
        if len(self.samples) == 0:
            raise RuntimeError(f"[{self.__class__.__name__}] nessun campione trovato in {self.label_file}")

    # ---------- API richieste alle sottoclassi ----------
    @abstractmethod
    def _load_labels(self) -> List[Dict[str, Any]]:
        """Ritorna list[{'image_path': Path, 'labels': Any}] per questo split."""
        ...

    @abstractmethod
    def get_labels_from_text_output(self, output: Any) -> Any:
        """Converte l'output testuale del VLM nelle etichette del dataset."""
        ...

    # ------------------------- PyTorch hooks -------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Ritorna (immagine, etichette).
        - immagine: PIL.Image RGB se transform=None, altrimenti il risultato della transform.
        - etichette: valore/dict come definito in _load_labels().
        """
        item = self.samples[idx]
        image_path = item.get("image_path")
        if not isinstance(image_path, Path):
            image_path = Path(image_path)

        if not image_path.exists():
            # se il CSV contiene path relativi, prova rispetto a image_folder
            alt = self.image_folder / image_path
            if alt.exists():
                image_path = alt
            else:
                raise FileNotFoundError(f"[{self.__class__.__name__}] immagine non trovata: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"[{self.__class__.__name__}] apertura immagine fallita ({image_path}): {e}")

        if self.transform is not None:
            image = self.transform(image)

        return image, item.get("labels")

    # --------------------------- Utility -----------------------------
    @staticmethod
    def get_available_datasets() -> list[str]:
        """Override per elencare dataset disponibili; default: lista vuota."""
        return []
    
    def get_train_class_counts(self, task: str) -> Optional[list[int]]:
        """
        Ritorna i conteggi per classe per il train, letti da train/class_counts.json.
        Regole:
        - Le chiavi sono stringhe di interi (classe). La chiave "-1" (unknown) viene ignorata.
        - Restituisce una lista di lunghezza (max_classe + 1), con 0 per le classi mancanti.
        - Se il file non esiste, il task non c’è o non ci sono classi valide → ritorna None.
        """
        counts_path = self.base_path / self.name / "train" / "class_counts.json"
        if not counts_path.exists():
            return None

        try:
            data = json.loads(counts_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        task_key = task.lower()
        raw = data.get(task_key)
        if not isinstance(raw, dict) or not raw:
            return None

        # Converte chiavi in int, ignora '-1'
        items = []
        for k, v in raw.items():
            try:
                idx = int(k)
                if idx >= 0:
                    items.append((idx, int(v)))
            except Exception:
                continue

        if not items:
            return None

        max_idx = max(i for i, _ in items)
        counts = [0] * (max_idx + 1)
        for i, c in items:
            counts[i] = int(c)

        return counts

    @property
    def samples_count(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, split={self.split!r}, N={len(self)})"