from transformers import PaliGemmaForConditionalGeneration
import torch
from PIL import Image
from .base_model import VLMModel

# NOTE:
# L'accesso al modello PaLI-Gemma richiede l'autenticazione Hugging Face.
# - Accetta le condizioni d'uso su: https://huggingface.co/google/paligemma-3b-mix-224
# - Genera un token su: https://huggingface.co/settings/tokens
# - Poi effettua il login una tantum oppure tramite variabile d'ambiente HF_TOKEN.

# ⚠️ Sicurezza: meglio NON hardcodare il token nel codice sorgente.
# Puoi usare: login(token=os.environ["HF_TOKEN"])
# oppure autenticarti una volta nella sessione interattiva.

from huggingface_hub import login
login(token="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")  # 🔐 SUGGERITO: spostare fuori da qui

class PaLIGemmaModel(VLMModel):
    """
    Implementazione del modello PaLI-Gemma per Vision-Language tasks.
    Estende VLMModel e utilizza un prompt che prevede il token speciale <image>.
    """

    def __init__(self, model_id=None, **model_kwargs):
        """
        Inizializza il modello PaLI-Gemma.

        Args:
            model_id (str, optional): ID del modello da Hugging Face.
                                      Se None, usa "google/paligemma-3b-mix-224" di default.
            model_kwargs: Parametri aggiuntivi per il caricamento del modello (es. quantization_config).
        """
        if model_id is None:
            model_id = "google/paligemma-3b-mix-224"
            # model_id = "google/paligemma2-3b-pt-448"  # alternativa con immagini 448x448
        super().__init__(model_id, **model_kwargs)

    def _load(self, **model_kwargs):
        """
        Carica il modello PaLI-Gemma e lo sposta sul device corretto.
        Chiama anche il metodo della superclasse per caricare il processor.
        """
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, **model_kwargs
        ).eval().to(self.device)
        super()._load(**model_kwargs)

    def generate_text(self, image: Image.Image, prompt: str, max_tokens: int = 100) -> str:
        """
        Genera una risposta a partire da un prompt testuale e un'immagine.

        Args:
            image (PIL.Image.Image): Immagine di input.
            prompt (str): Testo della domanda o del comando.
            max_tokens (int): Numero massimo di token da generare.

        Returns:
            str: Risposta generata dal modello.
        """
        # Il token <image> è richiesto da PaLI-Gemma per indicare la posizione visiva nel prompt.
        prompt = f"<image> {prompt}"
        return super().generate_text(image, prompt, max_tokens=max_tokens)