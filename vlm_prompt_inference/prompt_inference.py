# Percorso di progetto
from dotenv import load_dotenv
import os, sys
from pathlib import Path

load_dotenv()
project_root = os.getenv("PYTHONPATH")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import torch
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from datasets_vlm.evaluate_dataset import Evaluator
from tqdm import tqdm
import yaml  # pip install pyyaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Simple inference script (YAML config)")
    parser.add_argument("--config", type=str, default="vlm_prompt_inference/config.yaml",
                        help="Path al file di configurazione YAML")
    args = parser.parse_args()

    cfg_path = os.path.join(project_root, args.config) if project_root else args.config
    cfg = load_config(cfg_path)

    # Parametri da YAML
    model_name   = cfg["model_name"]
    quantization = cfg["quantization"]
    dataset_name = cfg["dataset_name"]
    max_tokens   = int(cfg.get("max_tokens", 100))

    # Output dir
    output_dir = os.path.join(project_root or ".", f"vlm_prompt_inference/eval/{model_name}/{quantization}/{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Modello
    model = VLMModelFactory.create_model(model_name, model_id=None, device=device, quantization=quantization)

    # Dataset
    ds_cfg = cfg.get("dataset", {}) or {}
    base_path = ds_cfg.get("base_path", None)
    train     = bool(ds_cfg.get("train", False))
    transform = None
    dataset = DatasetFactory.create_dataset(dataset_name, base_path=base_path, train=train, transform=transform)

    # Prompt
    prompts = cfg.get("prompts", {}) or {}
    if dataset_name in prompts:
        prompt = prompts[dataset_name]
    elif dataset_name == "MiviaPar" and "MiviaPar" in prompts:
        prompt = prompts["MiviaPar"]
    else:
        prompt = prompts.get("face_dataset", "")
    if not prompt:
        raise ValueError("Nessun prompt trovato in config (sezione 'prompts').")

    # Salva copia della config usata (YAML)
    with open(os.path.join(output_dir, "used_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # Inferenza
    preds, gts = [], []
    print(f"Running inference on dataset: {dataset_name}")
    try:
        for image, label in tqdm(dataset, desc="Processing images", unit="image"):
            output = model.generate_text(image, prompt, max_tokens=max_tokens)
            parsed_output = dataset.get_labels_from_text_output(output)
            preds.append(parsed_output)
            gts.append(label)
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente. Eseguo valutazione con i dati raccolti finora...")

    if preds and gts:
        Evaluator.evaluate(preds, gts, output_dir, dataset_name=dataset_name)
    else:
        print("Nessun dato valutabile.")

if __name__ == "__main__":
    main()