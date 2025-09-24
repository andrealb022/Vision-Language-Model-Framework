import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base_dataset import BaseDataset
import random
from tqdm import tqdm


class FaceDataset(BaseDataset):
    """
    Dataset unificato per più dataset facciali (CelebA_HQ, FairFace, LFW, RAF-DB, UTKFace, VggFace2-Train/Test, ...).
    Etichette standardizzate: gender, age, ethnicity, emotion, identity.
    Struttura su disco: <base>/<dataset>/{train|val|test}/(images/, labels.csv)
    """

    SUPPORTED_DATASETS = [
        "CelebA_HQ", "FairFace", "LFW", "RAF-DB", "TestDataset", "UTKFace",
        "VggFace2-Test", "VggFace2-Train"
    ]

    # Mapping nominale per etnie (usa solo queste 4 classi)
    ETHNICITY_LABELS = {
        "caucasian latin": 0,
        "caucasian": 0,
        "african american": 1,
        "east asian": 2,
        "asian indian": 3,
    }

    EMOTION_LABELS = {
        "surprise": 0, "fear": 1, "disgust": 2, "happiness": 3,
        "sadness": 4, "anger": 5, "neutral": 6,
    }

    AGE_LABELS = {
        "0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
        "40-49": 5, "50-59": 6, "60-69": 7, "70+": 8,
    }

    def __init__(self, dataset_name: str, split: str = "train", base_path = None, transform=None, age_is_regression: bool = False):
        """
        Args:
            dataset_name: nome del dataset (sottocartella).
            split: 'train' | 'val' | 'test'.
            base_path: radice dei dataset; se None usa default della BaseDataset.
            transform: trasformazioni immagine.
            age_is_regression: True → l'età è regressione (float); False → classificazione (classi 0..8).
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' non supportato. Supportati: {sorted(self.SUPPORTED_DATASETS)}"
            )
        self.age_is_regression = age_is_regression
        super().__init__(dataset_name=dataset_name, split=split, base_path=base_path, transform=transform)

    @staticmethod
    def get_available_datasets() -> List[str]:
        """Ritorna la lista dei dataset supportati."""
        return FaceDataset.SUPPORTED_DATASETS

    # ------------------------- Caricamento etichette -------------------------
    def _load_labels(self) -> List[Dict[str, Any]]:
        """
        Legge labels.csv (colonne attese: Path, Gender, Age, Ethnicity, Facial Emotion, Identity)
        e costruisce una lista di dict: {'image_path': Path, 'labels': {...}} per lo split corrente.
        - 'Path' può essere relativo (consigliato) o assoluto; se relativo è risolto rispetto a images/.
        - Se l'estensione nel CSV è assente, si provano [.jpg, .jpeg, .png].
        """
        df = pd.read_csv(self.label_file)
        samples: List[Dict[str, Any]] = []

        # Normalizza nomi colonne (tollerante a maiuscole/spazi)
        df.columns = [c.strip() for c in df.columns]

        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Loading labels"):
            try:
                # Path relativo all'immagine
                relative_path = Path(row["Path"].replace("\\", "/"))

                # Rimuove il prefisso ridondante se presente
                if relative_path.parts and relative_path.parts[0] == self.base_path.name:
                    relative_path = Path(*relative_path.parts[1:])

                # Cerca immagine con estensione valida
                extensions = [".jpg", ".jpeg", ".png"]
                for ext in extensions:
                    test_path = (self.base_path / relative_path).with_suffix(ext)
                    if test_path.exists():
                        image_path = test_path
                        break
                else:
                    raise FileNotFoundError(f"[Errore] Immagine non trovata: {relative_path} ({extensions})")

                # ------- Parsing campi -------
                # Gender
                gender = self._to_int_safe(row.get("Gender"), default=-1)

                # Age → float base
                age_val = self._to_float_safe(row.get("Age"), default=-1.0)
                age_label = age_val if self.age_is_regression else self._age_float_to_class(age_val)

                # Ethnicity
                ethnicity = self._to_int_safe(row.get("Ethnicity"), default=-1)

                # Emotion
                emotion = self._to_int_safe(row.get("Facial Emotion"), default=-1)

                # Identity (stringa; -1 se NaN)
                identity = str(row.get("Identity")).strip() if pd.notna(row.get("Identity")) else "-1"

                labels = {
                    "gender": gender,
                    "age": age_label,
                    "ethnicity": ethnicity,
                    "emotion": emotion,
                    "identity": identity,
                }
                samples.append({"image_path": image_path, "labels": labels})

            except Exception as e:
                print(f"[WARN] Riga CSV {idx + 2}: salto il campione → {e}")
                continue

        return samples

    # ------------------------- Parsing output VLM -------------------------
    def get_labels_from_text_output(self, output: str) -> Dict[str, Any]:
        """
        Converte una stringa del VLM in etichette standardizzate.
        Formato atteso (tollerante a spazi/maiuscole): "Male, 27.5, Asian Indian, Happiness"
        """
        try:
            parts = [x.strip().lower() for x in str(output).split(",")]
            if len(parts) < 4:
                raise ValueError(f"Output incompleto (attesi 4 campi): '{output}'")

            gender_str, age_str, ethnicity_str, emotion_str = parts[:4]

            # Gender
            gender = 1 if "female" in gender_str else 0 if "male" in gender_str else -1

            # Age
            age_val = self._to_float_safe(age_str, default=-1.0)
            age_label = age_val if self.age_is_regression else self._age_float_to_class(age_val)

            # Ethnicity (matching fuzzy + fallback)
            if "asian" in ethnicity_str and "caucasian" not in ethnicity_str:
                if "indian" in ethnicity_str:
                    ethnicity = self.ETHNICITY_LABELS.get("asian indian", -1)
                elif "east" in ethnicity_str:
                    ethnicity = self.ETHNICITY_LABELS.get("east asian", -1)
                else:
                    ethnicity = random.choice([
                        self.ETHNICITY_LABELS["east asian"],
                        self.ETHNICITY_LABELS["asian indian"],
                    ])
            else:
                ethnicity = next(
                    (v for k, v in self.ETHNICITY_LABELS.items() if k in ethnicity_str),
                    -1
                )

            # Emotion
            emotion = next((v for k, v in self.EMOTION_LABELS.items() if k in emotion_str), -1)

            return {"gender": gender, "age": age_label, "ethnicity": ethnicity, "emotion": emotion}
        except Exception as e:
            print(f"[WARN] Parsing output VLM fallito: {e}")
            return {
                "gender": -1,
                "age": (-1.0 if self.age_is_regression else -1),
                "ethnicity": -1,
                "emotion": -1,
            }

    # ------------------------------- Helper -------------------------------
    @staticmethod
    def _to_int_safe(v: Any, default: int = -1) -> int:
        try:
            return int(v) if pd.notna(v) else default
        except Exception:
            return default

    @staticmethod
    def _to_float_safe(v: Any, default: float = -1.0) -> float:
        try:
            return float(v) if pd.notna(v) else default
        except Exception:
            return default

    def _age_float_to_class(self, age_val: float) -> int:
        """Mappa un'età float alla classe 0..8; -1 se sconosciuta/negativa."""
        if age_val < 0:
            return -1
        bounds = [2, 9, 19, 29, 39, 49, 59, 69, float("inf")]
        for idx, upper in enumerate(bounds):
            if age_val <= upper:
                return idx
        return -1