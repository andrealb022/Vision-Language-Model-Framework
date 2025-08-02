from .blip2 import BLIP2OptModel
from .llava import LLaVAModel
from .paligemma import PaLIGemmaModel
from .base_model import VLMModel

class VLMModelFactory:
    """
    Factory per creare istanze di modelli Vision-Language (VLM).
    Permette di astrarre la logica di inizializzazione dei modelli tramite un nome simbolico.
    """

    _registry = {
        "blip2": BLIP2OptModel,
        "llava": LLaVAModel,
        "paligemma": PaLIGemmaModel,
    }

    @staticmethod
    def create_model(model_name: str, model_id: str, **model_kwargs) -> VLMModel:
        """
        Crea un'istanza del modello VLM richiesto.

        Args:
            model_name (str): Nome simbolico del modello (e.g. "blip2", "llava", "paligemma").
            model_id (str): Identificativo del modello da HuggingFace o repository custom.
            model_kwargs: Altri parametri da passare al costruttore del modello.

        Returns:
            VLMModel: Istanza del modello selezionato.

        Raises:
            ValueError: Se il nome del modello non è registrato nella factory.
        """
        model_name = model_name.lower()
        if model_name not in VLMModelFactory._registry:
            raise ValueError(f"Modello '{model_name}' non trovato. "
                             f"Disponibili: {list(VLMModelFactory._registry.keys())}")
        return VLMModelFactory._registry[model_name](model_id=model_id, **model_kwargs)

    @staticmethod
    def get_available_models():
        """
        Restituisce l'elenco dei modelli registrati nella factory.

        Returns:
            list: Lista dei nomi dei modelli disponibili.
        """
        return list(VLMModelFactory._registry.keys())


# Esempio di utilizzo:
if __name__ == "__main__":
    # Crea un'istanza del modello BLIP-2 con modello di default
    model = VLMModelFactory.create_model("blip2", model_id="Salesforce/blip2-opt-6.7b")

    # Stampa i modelli disponibili nella factory
    print("Modelli disponibili:", VLMModelFactory.get_available_models())