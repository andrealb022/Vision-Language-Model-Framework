import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration
from .base_model import VLMModel

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
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )

        return super().generate_text(image, prompt, max_tokens=max_tokens)

    def get_image_features(self, image: Image.Image, strategy: str = "mean") -> torch.Tensor:
        """
        Estrae un embedding visivo globale usando direttamente la vision_tower di LLaVA.

        Args:
            image (PIL.Image.Image): Immagine di input.
            strategy (str): Strategia di pooling: "mean" oppure "cls"

        Returns:
            torch.Tensor: Embedding visivo [B, D]
        """
        # SI PUO FARE ANCHE CON IMAGE_FEATURES
        # Preprocessamento immagine
        inputs = self.processor(
            images=image,
            text="Describe this image",  # richiesto anche se ignorato
            return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(self.device)

        # Sposta vision_tower su device
        self.model.vision_tower.to(self.device)
        self.model.multi_modal_projector.to(self.device)
        # Estrazione feature raw dal vision encoder (es. CLIP-ViT)
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