import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration
from .base_model import VLMModel
from .base_vision_backbone import VisionBackbone

class LLaVAModel(VLMModel):
    """
    Implementazione del modello LLaVA (Large Language and Vision Assistant),
    Estende la classe astratta VLMModel.
    """

    def __init__(self, model_id, device, quantization):
        """
        Inizializza il modello LLaVA.

        Args:
            model_id: ID del modello da HuggingFace. 
            device: Device su cui caricare il modello (es. "cuda" o "cpu").
            quantization: Tipo di quantizzazione da utilizzare (es. "fp32", "fp16", "8bit", "4bit").
        """
        if model_id is None:
            model_id = "llava-hf/llava-1.5-7b-hf"
        super().__init__(model_id, device, quantization)

    def _load_model(self):
        """
        Carica il modello LLaVA.
        Il processor viene gestito dalla superclasse.
        """
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, **self.get_model_kwargs()).eval()

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
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        return super().generate_text(image, prompt, max_tokens=max_tokens)

    def get_vision_backbone(self):
        """
        Ritorna un backbone visivo uniforme (VisionBackbone) per il probing.

        Returns:
            VisionBackbone: adapter che produce embedding globali [B, D].
        """
        return LLaVABackbone(self.processor, self.model.vision_tower, 1024, self.device)

################## BACKBONE PER ESTRAZIONE DI FEATURES ##################
class LLaVABackbone(VisionBackbone):
    """
    Adapter per LLaVA che estrae feature raw dal backbone visivo (CLIP).
    
    - Usa `vision_tower` (un `CLIPVisionModel`) per ottenere i token per-patch
      tramite `last_hidden_state` con shape [B, N, D].
    - Nei checkpoint LLaVA il `pooler_output` del vision encoder è in genere assente (None):
      per ottenere un embedding globale [B, D] facciamo pooling manuale.
    - Il pooling può essere:
        * "cls": prende il token [CLS] (indice 0) → valido per CLIP.
        * "mean": media sui token (robusto e generalizzabile).
    """
    def __init__(self, processor, vision_model, output_dim, device):
        super().__init__(processor, vision_model, output_dim, device)

    def forward(self, images, strategy: str = "cls"):
        """
        Args:
            images: PIL.Image o List[PIL.Image].
            strategy: "cls" (default) oppure "mean".
        Returns:
            torch.Tensor: embedding globali [B, D] sul device della backbone.
        """
        inputs = self.processor(images=images, text="Describe this image", return_tensors="pt").to(self.device)  # sposta su device
        # Estrazione feature raw dal vision encoder (CLIP)
        with torch.no_grad():
            image_embeds = self.vision_model(inputs["pixel_values"]).last_hidden_state  # [B, N, D]
        # Pooling: CLS token o media
        if strategy == "cls":
            return image_embeds[:, 0]  # [B, D]
        elif strategy == "mean":
            return image_embeds.mean(dim=1)  # [B, D]
        else:
            raise ValueError(f"Strategia pooling '{strategy}' non supportata")