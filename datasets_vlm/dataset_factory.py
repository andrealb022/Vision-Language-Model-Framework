from .mivia_par_dataset import MiviaParDataset
from .face_dataset import FaceDataset
from typing import Iterable, List, Dict
from torch.utils.data import ConcatDataset

class DatasetFactory:
    """
    Factory statica per creare istanze di dataset a partire dal nome.

    Mantiene un registro tra nomi dei dataset (stringhe) e classi concrete che estendono BaseDataset.
    Il registro è popolato dinamicamente interrogando ogni classe disponibile tramite
    il metodo `get_available_datasets()`.

    Attributi:
        _dataset_registry (dict): Mappa nome_dataset -> classe_dataset
    """

    _dataset_registry = {}

    # Lista di classi dataset supportate (da estendere)
    _registered_dataset_classes = [MiviaParDataset, FaceDataset]
    
    # Mappa task -> lista di dataset supportati
    TASK_TO_DATASETS: Dict[str, List[str]] = {
        "gender":    ["CelebA_HQ", "FairFace", "MiviaGender", "RAF-DB", "VggFace2-Train"],
        "ethnicity": ["FairFace"],
        "emotion":   ["RAF-DB"],
        "age":       ["VggFace2-Train", "FairFace"],
    }

    # Popolamento dinamico del registro (nome -> classe)
    for dataset_cls in _registered_dataset_classes:
        if hasattr(dataset_cls, "get_available_datasets"):
            for name in dataset_cls.get_available_datasets():
                _dataset_registry[name] = dataset_cls

    @staticmethod
    def create_dataset(dataset_name, base_path=None, train=False, transform=None, **kwargs):
        """
        Crea un'istanza del dataset specificato dal nome.

        Args:
            dataset_name (str): Nome simbolico del dataset (es. "MiviaPar", "UTKFace").
            base_path (str, optional): Percorso di base per il dataset.
            train (bool): Indica se il dataset è in modalità di addestramento o test.
            transform (callable, optional): Funzione di trasformazione da applicare ai campioni.

        Returns:
            BaseDataset: Istanza del dataset richiesto.

        Raises:
            ValueError: Se il nome non è presente nel registro dei dataset.
        """
        if dataset_name not in DatasetFactory._dataset_registry:
            available_datasets = DatasetFactory.get_available_datasets()
            raise ValueError(f"Dataset '{dataset_name}' non trovato. "
                             f"Dataset disponibili: {available_datasets}")
        
        dataset_class = DatasetFactory._dataset_registry[dataset_name]
        return dataset_class(dataset_name=dataset_name, base_path=base_path, train=train, transform=transform, **kwargs)

    @staticmethod
    def get_available_datasets():
        """
        Restituisce i nomi dei dataset disponibili nella factory.

        Returns:
            list: Lista di stringhe dei nomi registrati.
        """
        return list(DatasetFactory._dataset_registry.keys())

    @staticmethod
    def create_task_dataset(tasks: Iterable[str], base_path=None, train: bool = False, transform=None, **kwargs) -> ConcatDataset:
        """
        Crea un ConcatDataset con l'unione (deduplicata) dei dataset necessari
        per i task richiesti. Niente filtro delle label.
        """
        tasks = {t.lower() for t in tasks}
        valid = set(DatasetFactory.TASK_TO_DATASETS.keys())
        unknown = tasks - valid
        if unknown:
            raise ValueError(f"Task non supportati: {sorted(unknown)}. "
                             f"Task validi: {sorted(valid)}")

        # Unione con ordine dichiarato + deduplica sempre attiva
        selected_names: List[str] = []
        seen = set()
        for t in DatasetFactory.TASK_TO_DATASETS:
            if t in tasks:
                for name in DatasetFactory.TASK_TO_DATASETS[t]:
                    if name not in seen:
                        seen.add(name)
                        selected_names.append(name)

        if not selected_names:
            raise ValueError("Nessun dataset selezionato dai task forniti.")
        else:
            print(f"[Info] Dataset selezionati per i task {sorted(tasks)}: {selected_names}")

        # Istanzia e concatena
        instantiated = []
        for name in selected_names:
            if name not in DatasetFactory._dataset_registry:
                raise ValueError(
                    f"Il dataset '{name}' richiesto dai task {sorted(tasks)} non è registrato."
                )
            ds = DatasetFactory.create_dataset(
                name, base_path=base_path, train=train, transform=transform, **kwargs
            )
            instantiated.append(ds)

        return ConcatDataset(instantiated)


# Esempio di utilizzo:
if __name__ == "__main__":
    # Mostra i dataset disponibili
    print("Dataset disponibili:", DatasetFactory.get_available_datasets())

    # Crea un dataset
    dataset = DatasetFactory.create_dataset("RAF-DB", train=True)

    # Crea un dataset per task multipli (Gender + age)
    ds_gender_age = DatasetFactory.create_task_dataset(["gender", "age"], train=True)