from abc import ABC, abstractmethod
from PIL import Image
import torch
from transformers import AutoProcessor, BitsAndBytesConfig

class VLMModel(ABC):
    """
    Classe astratta base per modelli Vision-Language (VLM).

    Attributi:
        model_id (str): Identificatore del modello da caricare (HuggingFace o altro).
        device (torch.device): Dispositivo su cui eseguire il modello.
        quantization: Quantizzazione del modello.
        processor (AutoProcessor): Processor per la preparazione di immagini e prompt testuali.
    """

    def __init__(self, model_id, device, quantization):
        """
        Inizializza un modello VLM specificando il suo ID e eventuali parametri di configurazione.

        Args:
            model_id (str): ID del modello da caricare.
            device (torch.device): Dispositivo su cui eseguire il modello.
            quantization (str): Tipo di quantizzazione da utilizzare (es. "fp32", "fp16", "8bit", "4bit").
        """
        self.model_id = model_id
        self.device = device
        self.quantization = quantization

        # Caricamento del processor
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        print(f"Processor per il modello {self.model_id} caricato con successo.")
        
        # Caricamento del modello
        self._load_model()
        print(f"Modello {self.model_id} caricato con successo.")

    @abstractmethod
    def _load_model(self):
        """
        Carica il modello VLM specificato.
        """
        pass

    @abstractmethod
    def get_vision_backbone(self, cleanup: bool):
        """
        Ritorna il backbone visivo per l'estrazione delle caratteristiche.

        Returns:
            VisionBackbone: Il backbone visivo utilizzato dal modello.
        """
        pass

    def generate_text(self, image: Image.Image, prompt: str, max_tokens: int):
        """
        Genera testo dato un'immagine e un prompt testuale.

        Args:
            image (PIL.Image.Image): Immagine di input.
            prompt (str): Prompt testuale da associare all'immagine.
            max_tokens (int): Numero massimo di token da generare.

        Returns:
            str: Testo generato dal modello.
        """
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

            # Rimuove i token di input dal risultato generato per ottenere solo la risposta
            response = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]

        return self.processor.batch_decode(response, skip_special_tokens=True)[0].strip()

    def get_model_kwargs(self):
        """
        Restituisce i parametri per from_pretrained in base alla quantizzazione richiesta.
        """
        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            #"use_flash_attention_2": True,
        }
        if self.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif self.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif self.quantization == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        return model_kwargs