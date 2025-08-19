import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration
from .base_model import VLMModel

class BLIP2OptModel(VLMModel):
    """
    Implementazione del modello BLIP-2 con decoder OPT.
    Estende la classe astratta VLMModel.
    """

    def __init__(self, model_id, device, quantization):
        """
        Inizializza il modello BLIP-2 OPT.

        Args:
            model_id: ID del modello da caricare da HuggingFace.
            device: Device su cui caricare il modello (es. "cuda" o "cpu").
            quantization: Tipo di quantizzazione da utilizzare (es. "fp32", "fp16", "8bit", "4bit").
        """
        if model_id is None:
            model_id = "Salesforce/blip2-opt-6.7b"  # Modello di default se non specificato
        super().__init__(model_id, device, quantization)

    def _load_model(self):
        """
        Carica il modello BLIP2 con decoder OPT.
        Il processor viene gestito dalla superclasse.
        """
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id, **self.get_model_kwargs()).eval()

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

    def get_image_features(self, image: Image.Image):
        """
        Estrae le caratteristiche visive da un'immagine.

        Args:
            image (PIL.Image.Image): Immagine di input.

        Returns:
            torch.Tensor: Caratteristiche visive estratte.
        """
        # Preprocessing dell'immagine
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)  # sposta su device

        # Sposta vision_model sul device corretto
        self.model.vision_model = self.model.vision_model.to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)

        return vision_outputs.pooler_output