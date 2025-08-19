import pandas as pd
from .base_dataset import BaseDataset
import random

class MiviaParDataset(BaseDataset):
    """
    Dataset per il task MIVIA Person Attribute Recognition (PAR).

    Gestisce immagini annotate con attributi come colore degli abiti, genere,
    presenza di borsa e cappello. Le etichette vengono caricate da un file CSV senza intestazione.

    Attributi di output:
        - upper (colore della parte superiore)
        - lower (colore della parte inferiore)
        - gender (0=male, 1=female)
        - bag (0/1)
        - hat (0/1)
    """

    SUPPORTED_DATASETS = ["MiviaPar"]

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
        "yellow": 11
    }

    def __init__(self, dataset_name: str, base_path, train: bool, transform):
        """
        Inizializza il dataset MIVIA PAR.
        
        Args:
            dataset_name (str): Nome del dataset (usato come sottocartella).
            base_path (Path, optional): Percorso base. Default = ~/datasets_with_standard_labels/
            train (bool): Se True usa la partizione 'train/', altrimenti 'test/'.
            transform (callable, optional): Trasformazioni da applicare all'immagine (es. torchvision).
        """
        super().__init__(dataset_name=dataset_name, base_path=base_path, train=train, transform=transform)

    @staticmethod
    def get_available_datasets():
        """
        Restituisce i nomi dei dataset supportati dalla classe.

        Returns:
            list: Nomi dei dataset (per ora solo "MiviaPar").
        """
        return MiviaParDataset.SUPPORTED_DATASETS

    def _load_labels(self):
        """
        Carica le etichette dal file CSV, che non ha intestazione.
        Colonne attese: [path, upper, lower, gender, bag, hat]

        Returns:
            list of dict: Ogni elemento contiene:
                - image_path: Path dell'immagine
                - labels: dizionario con etichette numeriche
        """
        column_names = ["path", "upper", "lower", "gender", "bag", "hat"]
        df = pd.read_csv(self.label_file, header=None, names=column_names)

        samples = []
        for idx, row in df.iterrows():
            try:
                image_path = self.image_folder / row["path"]
                labels = {
                    "upper": int(row["upper"]),
                    "lower": int(row["lower"]),
                    "gender": int(row["gender"]),
                    "bag": int(row["bag"]),
                    "hat": int(row["hat"]),
                }
                samples.append({"image_path": image_path, "labels": labels})
            except Exception as e:
                print(f"[Errore] Parsing riga {idx + 1}: {e}")
                continue

        return samples

    def get_labels_from_text_output(self, output):
        """
        Converte una stringa generata da un VLM in etichette numeriche.

        Formato atteso:
            "Black, Black, Male, No, Yes"

        Returns:
            dict: Etichette nel formato MIVIA PAR standard:
                {
                    "upper": int (colore),
                    "lower": int (colore),
                    "gender": int (0=male, 1=female, -1=unknown),
                    "bag": int (0/1),
                    "hat": int (0/1)
                }
        """
        try:
            parts = [x.strip().lower() for x in output.split(",")]

            if len(parts) < 5:
                raise ValueError(f"[Errore] Output incompleto, attesi 5 valori: {output}")

            # Parsers locali
            def parse_color(color_str):
                for name, idx in self.COLOR_LABELS.items():
                    if name in color_str:
                        return idx
                return random.choice(list(self.COLOR_LABELS.values()))  # fallback random

            def parse_binary(value):
                if "yes" in value:
                    return 1
                if "no" in value:
                    return 0
                return -1

            def parse_gender(value):
                if "female" in value:
                    return 1
                if "male" in value:
                    return 0
                return -1

            # Parsing effettivo
            upper = parse_color(parts[0])
            lower = parse_color(parts[1])
            gender = parse_gender(parts[2])
            bag = parse_binary(parts[3])
            hat = parse_binary(parts[4])

            return {
                "upper": upper,
                "lower": lower,
                "gender": gender,
                "bag": bag,
                "hat": hat
            }

        except Exception as e:
            print(f"[Errore] Parsing output fallito: {e}")
            return {
                "upper": -1,
                "lower": -1,
                "gender": -1,
                "bag": -1,
                "hat": -1
            }