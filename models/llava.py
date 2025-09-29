import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration
from .base_model import VLMModel
from .vision_backbone import VisionBackbone
import re

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

    def get_vision_backbone(self, cleanup=True):
        """
        Ritorna la backbone visiva.
        Se cleanup=True, conserva solo la vision_tower e rimuove tutto il resto.
        """
        backbone = LLaVABackbone(self.processor, self.model.vision_tower, 1024, self.device)
        if cleanup:
            # Mantieni solo vision_tower, rimuovi gli altri sottocomponenti
            for name in list(self.model._modules.keys()):
                if name != "vision_tower":
                    #print(f"Pulizia del modulo: {name}")
                    delattr(self.model, name)

            # Sgancia riferimenti pesanti: il backbone ha già visual_tower + processor
            self.model = None
            self.processor = None
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return backbone

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

    def forward(self, images, strategy: str = "mean"):
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

    def unfreeze_last_k_layers(
        self,
        k: int = 2,
        parts: str = "all",               # "all" | "attn" | "mlp"
        include_embeddings: bool = True   # sblocca embeddings/norm globali/proj
    ):
        """
        Sblocca i parametri negli ultimi k layer dell'encoder visivo (self.vision_model).
        - Non congela gli altri parametri.
        - Le LayerNorm dei layer selezionati sono sempre incluse.
        - Non ritorna nulla; stampa un riepilogo finale.
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
            attn_hits = (".self_attn.q_proj" in param_name or
                        ".self_attn.k_proj" in param_name or
                        ".self_attn.v_proj" in param_name or
                        ".self_attn.out_proj" in param_name)
            mlp_hits = (".mlp.fc1" in param_name or ".mlp.fc2" in param_name)
            norm_hits = (".layer_norm" in param_name or ".ln" in param_name)  # sempre incluse

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
                "embeddings",            # patch/pos/class embeddings
                "pre_", "post_",         # pre/post layernorm (varia per modello)
                "proj",                  # projection heads
                "final_layer_norm",
            )
            for n, p in self.vision_model.named_parameters():
                if any(k in n for k in extra_keys):
                    p.requires_grad = True

        # 5) stampa riepilogo
        print(f"[unfreeze_last_k_layers] Scongelati {len(selected)} layer (indici: {sorted(selected)})")

    
    def get_lora_target_names(self, strategy):
        """
        strategy es.: {"last_k": 2, "attn_only": True}
        Ritorna i nomi (relativi alla backbone) dei nn.Linear negli ultimi K layer del CLIP encoder.
        """
        last_k = int(strategy.get("last_k", 2))
        attn_only = bool(strategy.get("attn_only", True))

        # trova gli indici layer: ...encoder.layers.<idx>...
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
                return (
                    "self_attn.q_proj" in n or
                    "self_attn.k_proj" in n or
                    "self_attn.v_proj" in n or
                    "self_attn.out_proj" in n
                )
            else:
                return (
                    "self_attn.q_proj" in n or
                    "self_attn.k_proj" in n or
                    "self_attn.v_proj" in n or
                    "self_attn.out_proj" in n or
                    "mlp.fc1" in n or
                    "mlp.fc2" in n
                )

        return self._find_linear(is_target)