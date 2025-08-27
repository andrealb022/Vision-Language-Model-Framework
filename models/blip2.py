import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration
from .base_model import VLMModel
from .base_vision_backbone import VisionBackbone

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

    def get_vision_backbone(self, cleanup=True):
        """
        Ritorna un backbone visivo (VisionBackbone) per il probing.
        Se cleanup=True, rimuove i riferimenti pesanti dal VLM per risparmiare memoria.
        """
        backbone = BLIP2Backbone(self.processor, self.model.vision_model, 1408, self.device)
        if cleanup:
            # Mantieni solo vision_model, rimuovi gli altri sottocomponenti
            for name in list(self.model._modules.keys()):
                if name != "vision_model":
                    print(f"Pulizia del modulo: {name}")
                    delattr(self.model, name)

            # Sgancia riferimenti pesanti: il backbone ha già visual_model + processor
            self.model = None
            self.processor = None
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return backbone

################## BACKBONE PER ESTRAZIONE DI FEATURES ##################
class BLIP2Backbone(VisionBackbone):
    """
    Adapter per BLIP-2:
    - Usa il vision encoder (self.vision_model) direttamente.
    - Restituisce l'embedding globale tramite `pooler_output` se disponibile.
    """
    def __init__(self, processor, vision_model, output_dim, device):
        super().__init__(processor, vision_model, output_dim, device)

    def forward(self, images):
        """
        Args:
            images (PIL.Image or List[PIL.Image]): immagini d'ingresso.

        Returns:
            torch.Tensor: embedding globali [B, D] sul device scelto.
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)  # sposta su device
        # Estrazione feature raw dal vision encoder (BLIP2)
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values=inputs["pixel_values"])
        return vision_outputs.pooler_output