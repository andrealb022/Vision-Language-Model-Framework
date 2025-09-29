import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration
from .base_model import VLMModel
from .vision_backbone import VisionBackbone
import re

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
                    #print(f"Pulizia del modulo: {name}")
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
    
    def unfreeze_last_k_layers(
        self,
        k: int = 2,
        parts: str = "all",              # "all" | "attn" | "mlp"
        include_embeddings: bool = True  # sblocca embeddings/norm globali/proj
        ):
        """
        Sblocca i parametri negli ultimi k layer dell'encoder visivo (self.vision_model).
        - Non congela gli altri parametri.
        """
        # 1) individua gli indici dei layer encoder
        layer_idxs = set()
        for n, _ in self.vision_model.named_modules():
            m = re.search(r"encoder\.layers\.(\d+)", n)
            if m:
                layer_idxs.add(int(m.group(1)))
        if not layer_idxs:
            raise RuntimeError("Non trovo 'encoder.layers.<idx>' in self.vision_model.")

        ordered = sorted(layer_idxs)
        selected = set(ordered[-int(k):]) if int(k) > 0 else set()

        # 2) helper di selezione
        def in_selected_layer(param_name: str) -> bool:
            m = re.search(r"encoder\.layers\.(\d+)", param_name)
            return bool(m) and int(m.group(1)) in selected

        def want(param_name: str) -> bool:
            if not in_selected_layer(param_name):
                return False
            # BLIP-2: attenzione tipicamente 'self_attn.qkv' + 'self_attn.projection'
            attn_hits = (
                ".self_attn.qkv" in param_name or
                ".self_attn.projection" in param_name
            )
            mlp_hits = (".mlp.fc1" in param_name or ".mlp.fc2" in param_name)
            # LayerNorm sempre incluse (diverse varianti di naming)
            norm_hits = (".layer_norm" in param_name or ".ln" in param_name or ".norm" in param_name)

            if parts == "all":
                return True
            if parts == "attn":
                return attn_hits or norm_hits
            if parts == "mlp":
                return mlp_hits or norm_hits
            return False

        # 3) sblocca parametri selezionati (senza congelare gli altri)
        for n, p in self.vision_model.named_parameters():
            if want(n):
                p.requires_grad = True

        # 4) opzionale: embeddings / norm/proj globali
        if include_embeddings:
            extra_keys = (
                "embeddings",            # naming generico
                "patch_embed",           # ViT-like
                "pos_embed", "position_embedding",
                "pre_", "post_",         # pre/post layernorm (varia per modello)
                "proj", "projection",    # projection heads
                "final_layer_norm",
            )
            for n, p in self.vision_model.named_parameters():
                if any(k in n for k in extra_keys):
                    p.requires_grad = True

        # 5) stampa riepilogo
        print(f"[unfreeze_last_k_layers] Scongelati {len(selected)} layer (indici: {sorted(selected)})")

    
    def get_lora_target_names(self, strategy):
        """
        strategy: {"last_k": 2, "attn_only": True}
        Ritorna i nomi (relativi alla backbone) dei Linear negli ultimi K layer.
        """
        last_k = int(strategy.get("last_k", 2))
        attn_only = bool(strategy.get("attn_only", True))

        # match '...encoder.layers.<idx>...'
        layer_indices = []
        for name, _ in self.named_modules():
            m = re.search(r"encoder\.layers\.(\d+)", name)
            if m:
                layer_indices.append(int(m.group(1)))
        if not layer_indices:
            return []

        max_idx = max(layer_indices)
        selected = set(range(max(0, max_idx - last_k + 1), max_idx + 1))

        def is_target(n: str) -> bool:
            m = re.search(r"encoder\.layers\.(\d+)", n)
            if not m or int(m.group(1)) not in selected:
                return False
            if attn_only:
                return ("self_attn.qkv" in n) or ("self_attn.projection" in n)
            else:
                return (
                    "self_attn.qkv" in n or "self_attn.projection" in n
                    or "mlp.fc1" in n or "mlp.fc2" in n
                )

        return self._find_linear(is_target)