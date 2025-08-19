import os
from io import BytesIO
import torch
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import PaliGemmaForConditionalGeneration
from .base_model import VLMModel

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

    def get_image_features(self, image: Image.Image, strategy: str = "cls"):
        """
        Estrae le caratteristiche visive da un'immagine.

        Args:
            image (PIL.Image.Image): Immagine di input.

        Returns:
            torch.Tensor: Caratteristiche visive estratte.
        """
        self.model.to(self.device)
        inputs = self.processor(images=image, text="<image> Describe this image", return_tensors="pt").to(self.device)
        pixel_values = inputs['pixel_values']
        with torch.no_grad():
            image_embeds = self.model.vision_tower(pixel_values).last_hidden_state  # [B, N, D]
            # Applica il multi-modal projector (mapping -> lingua)
            image_embeds = self.model.multi_modal_projector(image_embeds)  # [B, N, D']
            print(image_embeds)
        # Pooling: CLS token o media
        if strategy == "cls":
            return image_embeds[:, 0]  # [B, D']
        elif strategy == "mean":
            return image_embeds.mean(dim=1)  # [B, D']
        else:
            raise ValueError(f"Strategia pooling '{strategy}' non supportata")

if __name__ == "__main__":
    def load_sample_image():
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image

    model = PaLIGemmaModel()
    image = load_sample_image()
    features = model.get_image_features(image)

    print("✔️ Features estratte con successo!")
    print("Shape:", features.shape)
    print("Tipo:", type(features))