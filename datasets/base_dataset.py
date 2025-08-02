from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

# Percorso base di default per i dataset organizzati in cartelle standard
BASE_PATH = Path("~/datasets_with_standard_labels/").expanduser()
IMAGES_DIR = "images"
LABELS_FILE = "labels.csv"

class BaseDataset(Dataset, ABC):
    """
    Classe base astratta per dataset con immagini e etichette, strutturati su disco in cartelle.
    Compatibile con PyTorch DataLoader.

    Struttura attesa:
        base_path/
        └── dataset_name/
            ├── train/
            │   ├── images/
            │   └── labels.csv
            └── test/
                ├── images/
                └── labels.csv

    Le sottoclassi devono implementare:
    - `_load_labels()`: per caricare le etichette dal CSV in una lista di dict con 'image_path' e 'labels'
    - `get_labels_from_text_output()`: per estrarre le etichette a partire dall'output testuale di un VLM
    """

    def __init__(self, dataset_name: str, base_path: Path = None, train: bool = False, transform=None):
        """
        Inizializza il dataset.

        Args:
            dataset_name (str): Nome del dataset (usato come sottocartella).
            base_path (Path, optional): Percorso base. Default = ~/datasets_with_standard_labels/
            train (bool): Se True usa la partizione 'train/', altrimenti 'test/'.
            transform (callable, optional): Trasformazioni da applicare all'immagine (es. torchvision).
        """
        self.transform = transform
        self.base_path = Path(base_path).expanduser() if base_path else BASE_PATH
        split = "train" if train else "test"

        self.dataset_path = self.base_path / dataset_name / split
        self.image_folder = self.dataset_path / IMAGES_DIR
        self.label_file = self.dataset_path / LABELS_FILE

        # Carica tutti i campioni dall'implementazione della sottoclasse
        self.samples = self._load_labels()

    @abstractmethod
    def _load_labels(self) -> list:
        """
        Deve restituire una lista di dict con:
            - "image_path": Path all'immagine
            - "labels": Etichetta o dizionario di etichette

        Returns:
            list of dict
        """
        pass

    def __len__(self) -> int:
        """
        Numero di campioni nel dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Recupera l'immagine e le etichette per l'indice dato.

        Args:
            idx (int): Indice del campione.

        Returns:
            tuple: (immagine trasformata, etichette)
        """
        item = self.samples[idx]
        image_path = item["image_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"[Errore] Immagine non trovata: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"[Errore] Impossibile aprire immagine {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, item["labels"]

    @staticmethod
    def get_available_datasets():
        """
        Metodo statico per elencare i dataset disponibili.
        Può essere ridefinito dalle sottoclassi per restituire una lista statica o dinamica.

        Returns:
            list: Nomi dei dataset disponibili.
        """
        return []

    @abstractmethod
    def get_labels_from_text_output(self, output):
        """
        Estrae le etichette a partire dall'output testuale del modello VLM.

        Args:
            output: Output grezzo del modello.

        Returns:
            dict o valore semplice: Etichette strutturate da confrontare con le ground truth.
        """
        pass