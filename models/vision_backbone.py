import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Callable

class VisionBackbone(nn.Module, ABC):
    """
    Interfaccia astratta/adapter per estrarre embedding globali [B, D] da diversi VLM.

    Obiettivo:
        Fornire una API uniforme per il linear probing (o altri head leggeri) a prescindere
        dal tipo di vision encoder sottostante (CLIP, SigLIP, ecc.).

    Attributi:
        processor   : Processor/tokenizer Hugging Face per la pipeline di preprocessing immagini.
        vision_model: Modulo del vision encoder (es. CLIPVisionModel, SiglipVisionModel, ...).
        output_dim  : Dimensione attesa dell'embedding D (int)
        device      : Dispositivo su cui eseguire il backbone (torch.device).
    """
    def __init__(self, processor, vision_model, output_dim, device):
        super().__init__()
        self.processor = processor
        self.vision_model = vision_model
        self.output_dim = output_dim
        self.device = device
        self.vision_model.to(self.device)

    @abstractmethod
    def forward(self, images):
        """
        Metodo astratto: deve essere implementato dalle sottoclassi.
        Args:
            images: PIL.Image o List[PIL.Image] (il formato esatto può dipendere dal processor).

        Returns:
            torch.Tensor: embedding globali con shape [B, D] sul device `self.device`.
        """
        pass

    @abstractmethod
    def unfreeze_last_k_layers(self, k, parts, include_embeddings):
        """
        Sblocca i parametri negli ultimi k layer dell'encoder visivo (self.vision_model).
        - Non congela gli altri parametri.
        Args:
            k (int): Numero di layer finali da sbloccare.
            parts (str): Parti del modello da considerare (es. all, encoder ecc.).
            include_embeddings (bool): Se True, considera anche il layer di embedding.
        """
        pass
    
    @abstractmethod
    def get_lora_target_names(self, strategy: Dict) -> List[str]:
        """
        Ritorna la lista di path relativi ai sotto-moduli *interni alla backbone*
        (quindi nomi come 'vision_model.encoder.layers.10.attn.q_proj', ecc.).
        Lo strato superiore (es. 'backbone.') lo aggiungerai tu quando costruisci
        i target per l'intero modello (es. LinearProbe).
        """
        pass
    
    def _find_linear(self, name_pred: Callable[[str], bool]) -> List[str]:
        out = []
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and name_pred(name):
                out.append(name)
        return out