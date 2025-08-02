from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from .base_model import VLMModel

class BLIP2OptModel(VLMModel):
    """
    Implementazione del modello BLIP-2 con decoder OPT, basata sulla classe astratta VLMModel.
    Questo modello permette di generare testo a partire da una immagine e una domanda (prompt).
    """

    def __init__(self, model_id=None, **model_kwargs):
        """
        Inizializza il modello BLIP-2 OPT.

        Args:
            model_id (str, optional): ID del modello da caricare da HuggingFace.
                                      Se None, viene usato "Salesforce/blip2-opt-6.7b".
            model_kwargs: Parametri aggiuntivi per il caricamento del modello (es. quantization_config).
        """
        if model_id is None:
            model_id = "Salesforce/blip2-opt-6.7b"  # Modello di default se non specificato
        super().__init__(model_id, **model_kwargs)

    def _load(self, **model_kwargs):
        """
        Carica il modello BLIP2 con decoder OPT e lo imposta in modalità evaluation.
        Chiama poi `_load` della superclasse per caricare il processore.
        """
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id, **model_kwargs
        ).eval().to(self.device)
        super()._load(**model_kwargs)

    def generate_text(self, image: Image.Image, prompt: str, max_tokens: int = 100) -> str:
        """
        Genera una risposta testuale dato un prompt in linguaggio naturale e un'immagine.

        Args:
            image (PIL.Image.Image): Immagine da usare come contesto visivo.
            prompt (str): Prompt testuale (ad es. una domanda).
            max_tokens (int): Numero massimo di token da generare.

        Returns:
            str: Testo generato dal modello (risposta).
        """
        # Formattazione del prompt secondo lo schema standard BLIP2
        prompt = f"Question: {prompt}. Answer:"
        return super().generate_text(image, prompt, max_tokens=max_tokens)
