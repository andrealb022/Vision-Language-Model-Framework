# Percorso di progetto
from dotenv import load_dotenv
import os, sys, json
from pathlib import Path
load_dotenv()   # carica variabili da .env
project_root = os.getenv("PYTHONPATH")  # aggiungi PYTHONPATH se definito
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import torch
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from datasets_vlm.evaluate_dataset import Evaluator
from tqdm import tqdm

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

def main():
    # Solo il percorso del file di config come argomento
    parser = argparse.ArgumentParser(description="Simple inference script (config-driven)")
    parser.add_argument("--config", type=str, default="inference/config.json", help="Path al file di configurazione JSON")
    args = parser.parse_args()
    cfg = load_config(os.path.join(project_root, args.config) if project_root else args.config)

    # Lettura parametri dalla config
    model_name   = cfg["model_name"]
    quantization = cfg["quantization"]
    dataset_name = cfg["dataset_name"]
    max_tokens   = int(cfg.get("max_tokens", 100))

    output_dir = os.path.join(project_root, f"inference/eval/{model_name}/{quantization}/{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Setup modello
    model_id = None
    model = VLMModelFactory.create_model(model_name, model_id, device, quantization)

    # Setup dataset
    ds_cfg = cfg.get("dataset", {})
    base_path = ds_cfg.get("base_path", None)
    train     = bool(ds_cfg.get("train", False))
    transform = None  # se vuoi, aggiungi gestione trasformazioni nella config
    dataset = DatasetFactory.create_dataset(dataset_name, base_path=base_path, train=train, transform=transform)

    # Scegli il prompt dalla config
    prompts = cfg.get("prompts", {})
    if dataset_name in prompts:
        prompt = prompts[dataset_name]
    elif dataset_name == "MiviaPar" and "MiviaPar" in prompts:
        prompt = prompts["MiviaPar"]
    else:
        prompt = prompts.get("face_dataset", "")

    if not prompt:
        raise ValueError("Nessun prompt trovato in config (chiave 'prompts').")

    # Salva una copia della config usata per riproducibilità
    with open(os.path.join(output_dir, "used_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Inizio inferenza
    preds, gts = [], []
    print(f"Running inference on dataset: {dataset_name}")
    try:
        for image, label in tqdm(dataset, desc="Processing images", unit="image"):
            output = model.generate_text(image, prompt, max_tokens=max_tokens)
            parsed_output = dataset.get_labels_from_text_output(output)
            preds.append(parsed_output)
            gts.append(label)
            # print(f"Output: {output}, Parsed: {parsed_output}, GT: {label}")
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente. Eseguo valutazione con i dati raccolti finora...")

    if preds and gts:
        Evaluator.evaluate(preds, gts, output_dir, dataset_name=dataset_name)
    else:
        print("Nessun dato valutabile.")

if __name__ == "__main__":
    main()