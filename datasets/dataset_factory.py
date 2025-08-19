from .mivia_par_dataset import MiviaParDataset
from .face_dataset import FaceDataset
from tqdm import tqdm

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

    # Popolamento dinamico del registro (nome -> classe)
    for dataset_cls in _registered_dataset_classes:
        if hasattr(dataset_cls, "get_available_datasets"):
            for name in dataset_cls.get_available_datasets():
                _dataset_registry[name] = dataset_cls

    @staticmethod
    def create_dataset(dataset_name, base_path=None, train=False, transform=None):
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
        return dataset_class(dataset_name=dataset_name, base_path=base_path, train=train, transform=transform)

    @staticmethod
    def get_available_datasets():
        """
        Restituisce i nomi dei dataset disponibili nella factory.

        Returns:
            list: Lista di stringhe dei nomi registrati.
        """
        return list(DatasetFactory._dataset_registry.keys())


# Esempio di utilizzo:
if __name__ == "__main__":
    # Mostra i dataset disponibili
    print("Dataset disponibili:", DatasetFactory.get_available_datasets())

    # Crea un dataset
    dataset = DatasetFactory.create_dataset("VggFace2-Train", train=True)