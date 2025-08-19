import pandas as pd
from pathlib import Path
from .base_dataset import BaseDataset
import random

class FaceDataset(BaseDataset):
    """
    Dataset unificato per più dataset facciali (CelebA_HQ, FairFace, LFW, ecc.).
    Supporta le etichette standardizzate: gender, age, ethnicity, emotion, identity.
    Compatibile con la factory e strutture standardizzate su disco.
    """

    # Dataset supportati dalla factory
    SUPPORTED_DATASETS = [
        "CelebA_HQ", "FairFace", "LFW", "MiviaGender",
        "RAF-DB", "TestDataset", "UTKFace", "VggFace2-Test","VggFace2-Train"
    ]

    # Mappatura nominale per etnie
    ETHNICITY_LABELS = {
        "caucasian latin": 0,
        "caucasian": 0,
        "african american": 1,
        "east asian": 2,
        "asian indian": 3
    }

    # Mappatura per emozioni facciali
    EMOTION_LABELS = {
        "surprise": 0,
        "fear": 1,
        "disgust": 2,
        "happiness": 3,
        "sadness": 4,
        "anger": 5,
        "neutral": 6
    }

    def __init__(self, dataset_name: str, base_path: Path, train: bool, transform):
        """
        Inizializza il dataset facciale.

        Args:
            dataset_name (str): Nome del dataset (usato come sottocartella).
            base_path (Path, optional): Percorso base. Default = ~/datasets_with_standard_labels/
            train (bool): Se True usa la partizione 'train/', altrimenti 'test/'.
            transform (callable, optional): Trasformazioni da applicare all'immagine (es. torchvision).
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"[Errore] Dataset '{dataset_name}' non supportato. "
                f"Supportati: {sorted(self.SUPPORTED_DATASETS)}"
            )
        else:
            super().__init__(dataset_name=dataset_name, base_path=base_path, train=train, transform=transform)

    @staticmethod
    def get_available_datasets():
        """
        Restituisce la lista dei dataset supportati dalla classe.

        Returns:
            list: Nomi dei dataset.
        """
        return FaceDataset.SUPPORTED_DATASETS

    def _load_labels(self):
        """
        Carica le etichette da un CSV nel formato standard:
        Path, Gender, Age, Ethnicity, Facial Emotion, Identity

        Returns:
            list of dict: ciascun elemento ha chiavi 'image_path' e 'labels'.
        """
        df = pd.read_csv(self.label_file)
        samples = []

        for idx, row in df.iterrows():
            try:
                # Path relativo all'immagine
                relative_path = Path(row["Path"].replace("\\", "/"))

                # Rimuove il prefisso ridondante se presente
                if relative_path.parts[0] == self.base_path.name:
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

                labels = {
                    "gender": int(row["Gender"]) if pd.notna(row["Gender"]) else -1,
                    "age": int(row["Age"]) if pd.notna(row["Age"]) else -1,
                    "ethnicity": int(row["Ethnicity"]) if pd.notna(row["Ethnicity"]) else -1,
                    "emotion": int(row["Facial Emotion"]) if pd.notna(row["Facial Emotion"]) else -1,
                    "identity": str(row["Identity"]) if pd.notna(row["Identity"]) else -1,
                }

                samples.append({"image_path": image_path, "labels": labels})
            except Exception as e:
                print(f"[Errore] Riga {idx + 2}: parsing fallito - {e}")
                continue

        return samples

    def get_labels_from_text_output(self, output):
        """
        Estrae le etichette da una stringa generata da un VLM.

        Esempio atteso:
            "Male, 27.5, Asian Indian, Happiness"

        Returns:
            dict: Etichette standardizzate nel formato:
                {
                    "gender": int (0=male, 1=female, -1=unknown),
                    "age": float,
                    "ethnicity": int (0–3, -1=unknown),
                    "emotion": int (0–6, -1=unknown)
                }
        """
        try:
            parts = [x.strip().lower() for x in output.split(",")]

            if len(parts) < 4:
                raise ValueError(f"[Errore] Output incompleto (attesi 4 campi): '{output}'")

            gender_str, age_str, ethnicity_str, emotion_str = parts

            # Gender
            gender = 1 if "female" in gender_str else 0 if "male" in gender_str else -1

            # Age
            try:
                age = float(age_str)
            except ValueError:
                age = -1

            # Ethnicity (gestione fuzzy + fallback)
            if "asian" in ethnicity_str and "caucasian" not in ethnicity_str:
                if "indian" in ethnicity_str:
                    ethnicity = self.ETHNICITY_LABELS.get("asian indian", -1)
                elif "east" in ethnicity_str:
                    ethnicity = self.ETHNICITY_LABELS.get("east asian", -1)
                else:
                    ethnicity = random.choice([
                        self.ETHNICITY_LABELS["east asian"],
                        self.ETHNICITY_LABELS["asian indian"]
                    ])
            else:
                ethnicity = next(
                    (v for k, v in self.ETHNICITY_LABELS.items() if k in ethnicity_str),
                    -1
                )

            # Emotion
            emotion = next(
                (v for k, v in self.EMOTION_LABELS.items() if k in emotion_str),
                -1
            )

            return {
                "gender": gender,
                "age": age,
                "ethnicity": ethnicity,
                "emotion": emotion,
            }

        except Exception as e:
            print(f"[Errore] Parsing output fallito: {e}")
            return {
                "gender": -1,
                "age": -1,
                "ethnicity": -1,
                "emotion": -1,
            }