from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from .base_dataset import BaseDataset


class MiviaParDataset(BaseDataset):
    """
    Dataset per il task MIVIA Person Attribute Recognition (PAR).

    Etichette di output (dizionarie per campione):
      - upper : colore parte superiore (int)   — mapping in COLOR_LABELS, -1 se sconosciuto
      - lower : colore parte inferiore (int)   — mapping in COLOR_LABELS, -1 se sconosciuto
      - gender: 0=male, 1=female, -1=unknown
      - bag   : 0/1, -1=unknown
      - hat   : 0/1, -1=unknown

    Struttura su disco (per split in {'train','val','test'}):
      <base>/<dataset_name>/<split>/{images/, labels.csv}

    Il CSV NON ha intestazione. Colonne attese:
      [path, upper, lower, gender, bag, hat]
    """

    SUPPORTED_DATASETS = ["MiviaPar"]

    # Colori (classi 1..11). -1 verrà usato come "unknown".
    COLOR_LABELS = {
        "black": 1, "dark": 1,
        "blue": 2,
        "brown": 3,
        "gray": 4,
        "green": 5,
        "orange": 6,
        "pink": 7,
        "purple": 8,
        "red": 9,
        "white": 10,
        "yellow": 11,
    }

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        base_path: Optional[Path] = None,
        transform=None,
    ):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' non supportato. Ammessi: {self.SUPPORTED_DATASETS}")
        super().__init__(dataset_name=dataset_name, split=split, base_path=base_path, transform=transform)

    @staticmethod
    def get_available_datasets() -> List[str]:
        """Elenco dei dataset supportati."""
        return MiviaParDataset.SUPPORTED_DATASETS

    # ------------------------- Caricamento etichette -------------------------
    def _load_labels(self) -> List[Dict[str, Any]]:
        """
        Legge labels.csv (senza header) e costruisce:
          [{'image_path': Path, 'labels': {...}}, ...]
        Path nel CSV può essere relativo a images/ (consigliato).
        """
        column_names = ["path", "upper", "lower", "gender", "bag", "hat"]
        df = pd.read_csv(self.label_file, header=None, names=column_names)
        df.columns = [c.strip() for c in df.columns]

        samples: List[Dict[str, Any]] = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"[{self.name}/{self.split}] Loading labels"):
            try:
                rel = str(row["path"]).strip().replace("\\", "/")
                image_path = self._resolve_image_path(rel)

                labels = {
                    "upper": self._color_to_id(row.get("upper")),
                    "lower": self._color_to_id(row.get("lower")),
                    "gender": self._to_int_safe(row.get("gender"), default=-1),
                    "bag": self._to_bin_safe(row.get("bag")),
                    "hat": self._to_bin_safe(row.get("hat")),
                }
                samples.append({"image_path": image_path, "labels": labels})
            except Exception as e:
                print(f"[WARN] Riga CSV {i + 1}: salto → {e}")
                continue

        if not samples:
            raise RuntimeError(f"Nessun campione valido in {self.label_file}")
        return samples

    # ------------------------- Parsing output VLM -------------------------
    def get_labels_from_text_output(self, output: str) -> Dict[str, int]:
        """
        Converte una stringa VLM in etichette numeriche.
        Formato atteso (case-insensitive, separato da virgole):
          "Black, Black, Male, No, Yes"
        """
        try:
            parts = [p.strip().lower() for p in str(output).split(",")]
            if len(parts) < 5:
                raise ValueError(f"Output incompleto (attesi 5 campi): {output}")

            upper = self._match_color(parts[0])
            lower = self._match_color(parts[1])
            gender = 1 if "female" in parts[2] else 0 if "male" in parts[2] else -1
            bag = self._parse_yesno(parts[3])
            hat = self._parse_yesno(parts[4])

            return {"upper": upper, "lower": lower, "gender": gender, "bag": bag, "hat": hat}
        except Exception as e:
            print(f"[WARN] Parsing output VLM fallito: {e}")
            return {"upper": -1, "lower": -1, "gender": -1, "bag": -1, "hat": -1}

    # ------------------------------- Helper -------------------------------
    def _resolve_image_path(self, rel_or_abs: str) -> Path:
        """Risolvi path immagine: se relativo → rispetto a images/; valida l'esistenza."""
        p = Path(rel_or_abs)
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"Immagine non trovata: {p}")
            return p
        # relativo: può includere sottocartelle
        candidate = self.image_folder / p
        if not candidate.exists():
            raise FileNotFoundError(f"Immagine non trovata (relativa): {candidate}")
        return candidate

    @staticmethod
    def _to_int_safe(v, default: int = -1) -> int:
        try:
            return int(v)
        except Exception:
            return default

    @staticmethod
    def _to_bin_safe(v) -> int:
        """Converte in 0/1/-1. Accetta 0/1, '0'/'1', 'yes'/'no' (case-insensitive)."""
        s = str(v).strip().lower()
        if s in {"1", "yes", "y", "true"}:
            return 1
        if s in {"0", "no", "n", "false"}:
            return 0
        try:
            return 1 if int(v) == 1 else 0 if int(v) == 0 else -1
        except Exception:
            return -1

    def _color_to_id(self, v) -> int:
        """
        Converte un colore (stringa o intero) nella classe colore:
          - se già intero → ritorna int(v)
          - se stringa → matching lessicale
          - altrimenti → -1
        """
        # già numerico?
        try:
            return int(v)
        except Exception:
            pass
        # stringa: match lessicale
        s = str(v).strip().lower()
        return self._match_color(s)

    def _match_color(self, s: str) -> int:
        """Trova l'id colore dalla stringa; -1 se nessun match."""
        for name, idx in self.COLOR_LABELS.items():
            if name in s:
                return idx
        return -1