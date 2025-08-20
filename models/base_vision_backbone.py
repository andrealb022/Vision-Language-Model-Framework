import torch
import torch.nn as nn
from abc import ABC, abstractmethod

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