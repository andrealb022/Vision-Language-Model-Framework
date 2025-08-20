import os
import torch
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import PaliGemmaForConditionalGeneration
from .base_model import VLMModel
from .base_vision_backbone import VisionBackbone

# -----------------------------------------------------------------------------
# Autenticazione (opzionale) a Hugging Face tramite variabile d'ambiente HF_TOKEN.
# - Accetta i ToS del modello: https://huggingface.co/google/paligemma-3b-mix-224
# - Crea un token: https://huggingface.co/settings/tokens
# - Esporta HF_TOKEN nel tuo ambiente o file .env
# -----------------------------------------------------------------------------
load_dotenv()  # Carica variabili da .env
_hf_token = os.getenv("HF_TOKEN")
login(token=_hf_token)

class PaLIGemmaModel(VLMModel):
    """
    Implementazione del modello PaLI-Gemma per Vision-Language tasks.
    Estende la classe astratta VLMModel.
    """

    def __init__(self, model_id, device, quantization):
        """
        Inizializza il modello PaLI-Gemma.

        Args:
            model_id: ID del modello su Hugging Face. Default: "google/paligemma-3b-mix-224".
            device: Device su cui caricare il modello (es. "cuda" o "cpu").
            quantization: Tipo di quantizzazione da utilizzare (es. "fp32", "fp16", "8bit", "4bit").
        """
        if model_id is None:
            model_id = "google/paligemma-3b-mix-224"
            # model_id = "google/paligemma2-3b-pt-448"  # alternativa con immagini 448x448
        super().__init__(model_id, device, quantization)

    def _load_model(self):
        """
        Carica il modello PaLI-Gemma.
        Il processor viene gestito dalla superclasse.
        """
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, **self.get_model_kwargs()).eval()

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

    def get_vision_backbone(self):
        """
        Ritorna un backbone visivo uniforme (VisionBackbone) per il probing.

        Returns:
            VisionBackbone: adapter che produce embedding globali [B, D].
        """
        return PaliGemmaBackbone(self.processor, self.model.vision_tower, 1152, self.device)

################## BACKBONE PER ESTRAZIONE DI FEATURES ##################
class PaliGemmaBackbone(VisionBackbone):
    """
    Adapter per PaLI‑Gemma (vision tower: SigLIP).

    - SigLIP non usa il token [CLS]; il vision encoder produce solo token per-patch.
    - Il `pooler_output` in genere non è definito (None) per ottenere un embedding
      globale [B, D] si applica mean pooling sui token.
    - `last_hidden_state` ha shape [B, N, D] (N = numero di patch).
    """
    def __init__(self, processor, vision_model, output_dim, device):
        super().__init__(processor, vision_model, output_dim, device)

    def forward(self, images):
        """
        Args:
            images: PIL.Image o List[PIL.Image].

        Returns:
            torch.Tensor: embedding globali [B, D] (mean pooling) sul device selezionato.
        """
        inputs = self.processor(images=images, text="<image> Describe this image", return_tensors="pt").to(self.device)
        # Estrazione feature raw dal vision encoder (SigLIP)
        with torch.no_grad():
            image_embeds = self.vision_model(inputs['pixel_values']).last_hidden_state  # [B, N, D]
        return image_embeds.mean(dim=1)  # [B, D]