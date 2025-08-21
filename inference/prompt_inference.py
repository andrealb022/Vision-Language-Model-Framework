# Percorso di progetto
from dotenv import load_dotenv
import os, sys
load_dotenv()   # carica variabili da .env
project_root = os.getenv("PYTHONPATH")  # aggiungi PYTHONPATH se definito
if project_root and project_root not in sys.path:
    sys.path.append(project_root)
import argparse
import torch
from pathlib import Path
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from datasets_vlm.evaluate import Evaluator
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference script")
    # Model arguments
    parser.add_argument("--model_name", type=str, default="llava", choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp16", choices=["4bit", "8bit", "fp16", "fp32"])
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="VggFace2-Test", choices=DatasetFactory.get_available_datasets())
    parser.add_argument("--train", type=bool, default=False, help="True -> dataset_train, False -> dataset_test")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Imposta il path di output relativo al project_root
    if project_root:
        output_dir = os.path.join(project_root, "eval", args.model_name, args.dataset_name, args.quantization)
    else:
        print("PYTHONPATH non definito. Utilizzo 'output' relativo alla directory corrente.")
        output_dir = os.path.join("eval", args.model_name, args.dataset_name, args.quantization)

    # Setup del modello
    model_id = None  # Placeholder for model ID if needed
    model = VLMModelFactory.create_model(args.model_name, model_id, device, args.quantization)
    max_tokens = 100
    
    # Setup del dataset
    base_path = None # Placeholder for base path if needed (altrimenti utilizza il base_path di default)
    transform = None # Placeholder for any transformations
    dataset = DatasetFactory.create_dataset(args.dataset_name, base_path=base_path, train=args.train, transform=transform)

    # Esegui inferenza
    if args.dataset_name == "MiviaPar":
        prompt = (
            "Analyze the person in this image and provide the following information in the exact format requested. "
            "Return your response as comma-separated values in this specific order:\n"
            "Color Upper Clothes,Color Lower Clothes,Gender,Presence of bag,Presence of hat\n"
            "Instructions:\n"
            "- Color Upper Clothes: Choose from: black, white, gray, red, blue, green, yellow, orange, purple, pink, brown\n"
            "- Color Lower Clothes: Choose from: black, white, gray, red, blue, green, yellow, orange, purple, pink, brown\n"
            "- Gender: male or female\n"
            "- Presence of bag: yes or no\n"
            "- Presence of hat: yes or no\n"
            "Example output format:\n"
            "blue,black,male,yes,no\n"
            "Important: Provide only the comma-separated values, no additional text or explanation."
        )
    else:
        prompt = (
            "Analyze the face in this image and provide the following information in the exact format requested. "
            "Return your response as comma-separated values in this specific order:\n"
            "Gender,Age,Ethnicity,Facial Emotion\n"
            "Instructions:\n"
            "- Gender: male or female\n"
            "- Age: Estimate age as a number (e.g., 25.0, 34.5)\n"
            "- Ethnicity: Choose from: caucasian, african american, east asian, asian indian\n"
            "- Facial Emotion: Choose from: anger, disgust, fear, happiness, sadness, surprise\n"
            "Example output format:\n"
            "male,28.5,east asian,happiness\n"
            "Important: Provide only the comma-separated values, no additional text or explanation."
        )

    # Inizio inferenza
    preds = []
    gts = []
    print(f"Running inference on dataset: {args.dataset_name}")
    try:
        for image, label in tqdm(dataset, desc="Processing images", unit="image"):
            output = model.generate_text(image, prompt, max_tokens=max_tokens)
            parsed_output = dataset.get_labels_from_text_output(output)
            preds.append(parsed_output)
            gts.append(label)
            #print(f"Output: {output}, Parsed: {parsed_output}, GT: {label}")
    
    except KeyboardInterrupt:
        print("\nInterrotto dall'utente. Eseguo valutazione con i dati raccolti finora...")

    if preds and gts:
        Evaluator.evaluate(preds, gts, output_dir, dataset_name=args.dataset_name)
    else:
        print("Nessun dato valutabile.")

if __name__ == "__main__":
    main()
