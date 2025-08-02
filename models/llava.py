from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
from .base_model import VLMModel

class LLaVAModel(VLMModel):
    """
    Implementazione del modello LLaVA (Large Language and Vision Assistant),
    basata su LLaVA 1.5 HuggingFace. Estende la classe astratta VLMModel.
    """

    def __init__(self, model_id=None, **model_kwargs):
        """
        Inizializza il modello LLaVA.

        Args:
            model_id (str, optional): ID del modello da HuggingFace. Se None, usa
                                      "llava-hf/llava-1.5-7b-hf" come default.
            model_kwargs: Parametri aggiuntivi per il caricamento del modello (es. device_map, quantization_config).
        """
        if model_id is None:
            model_id = "llava-hf/llava-1.5-7b-hf"
        super().__init__(model_id, **model_kwargs)

    def _load(self, **model_kwargs):
        """
        Carica il modello LLaVA e lo imposta in modalità evaluation.
        Chiama poi `_load` della superclasse per caricare il processore associato.
        """
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, **model_kwargs
        ).eval().to(self.device)
        super()._load(**model_kwargs)

    def generate_text(self, image: Image.Image, prompt: str, max_tokens: int = 100) -> str:
        """
        Genera una risposta testuale dato un'immagine e un prompt, utilizzando
        il formato conversazionale previsto da LLaVA.

        Args:
            image (PIL.Image.Image): Immagine di input.
            prompt (str): Prompt testuale dell'utente.
            max_tokens (int): Numero massimo di token da generare.

        Returns:
            str: Risposta testuale generata dal modello.
        """
        # Costruisce la conversazione in formato chat LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            },
        ]

        # Converte il formato chat in prompt lineare per il modello
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        return super().generate_text(image, prompt, max_tokens=max_tokens)