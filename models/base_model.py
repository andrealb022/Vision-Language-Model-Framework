from abc import ABC, abstractmethod
from PIL import Image
import torch
from transformers import AutoProcessor

class VLMModel(ABC):
    """
    Classe astratta base per modelli Vision-Language (VLM).
    Ogni sottoclasse concreta deve definire il modello e implementare eventuali metodi specifici.

    Attributi:
        model_id (str): Identificatore del modello da caricare (HuggingFace o altro).
        device (torch.device): Dispositivo su cui eseguire il modello.
        processor (AutoProcessor): Processore per la preparazione di immagini e prompt testuali.
    """

    def __init__(self, model_id, **model_kwargs):
        """
        Inizializza un modello VLM specificando il suo ID e eventuali parametri di configurazione.

        Args:
            model_id (str): ID del modello da caricare.
            model_kwargs: Argomenti opzionali per la configurazione o il caricamento del modello.
        """
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self._load(**model_kwargs)

    def _load(self, **model_kwargs):
        """
        Carica il processore associato al modello (per immagini e testo).
        Il modello vero e proprio deve essere caricato nelle sottoclassi concrete,
        che dovranno anche definire `self.model`.
        """
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        print(f"Processor for model {self.model_id} loaded successfully.")

    def generate_text(self, image: Image.Image, prompt: str, max_tokens: int):
        """
        Genera testo dato un'immagine e un prompt testuale.

        Args:
            image (PIL.Image.Image): Immagine di input.
            prompt (str): Prompt testuale da associare all'immagine.
            max_tokens (int): Numero massimo di token da generare.

        Returns:
            str: Testo generato dal modello.
        """
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

            # Rimuove i token di input dal risultato generato per ottenere solo la risposta
            response = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]

        return self.processor.batch_decode(response, skip_special_tokens=True)[0].strip()