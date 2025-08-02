import argparse
import torch
import json
from pathlib import Path
from transformers import BitsAndBytesConfig
from models.model_factory import VLMModelFactory
from datasets.dataset_factory import DatasetFactory
from evaluate import Evaluator
import os
from tqdm import tqdm

def get_model_kwargs(args):
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    elif args.quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    elif args.quantization == "fp16":
        model_kwargs["torch_dtype"] = torch.float16
    return model_kwargs

def get_dataset_kwargs(args):
    return {
        "train": args.train,
        "transform": None,  # Placeholder for any transformations
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Simple inference script")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    # Model arguments
    parser.add_argument("--model_name", type=str, default="llava", choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp32", choices=["4bit", "8bit", "fp16", "fp32"])
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="VggFace2-Test", choices=DatasetFactory.get_available_datasets())
    parser.add_argument("--train", type=bool, default=False, help="True -> dataset_train, False -> dataset_test")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir=os.path.join(args.output_dir, args.model_name, args.dataset_name, args.quantization)
    # Setup del modello
    model_kwargs = get_model_kwargs(args)
    model_id = None  # Placeholder for model ID if needed
    model = VLMModelFactory.create_model(args.model_name, model_id=model_id, **model_kwargs)
    max_tokens = 100
    # Setup del dataset
    dataset_kwargs = get_dataset_kwargs(args)
    dataset = DatasetFactory.create_dataset(args.dataset_name, **dataset_kwargs)
    
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
