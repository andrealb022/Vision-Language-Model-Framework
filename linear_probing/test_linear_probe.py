"""
Linear Probe - Test/Evaluation Script

Dataset e task disponibili per il test:
- CelebA_HQ       → gender
- FairFace        → gender, ethnicity
- LFW             → gender
- UTKFace         → gender, age, ethnicity
- MiviaGender     → gender
- RAF-DB          → gender, facial emotion
- VggFace2-Test   → gender, age, ethnicity
"""

from dotenv import load_dotenv
import os, sys
load_dotenv()
project_root = os.getenv("PYTHONPATH")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model_factory import VLMModelFactory
from linear_probing.linear_probe import LinearProbe
from datasets_vlm.dataset_factory import DatasetFactory
from datasets_vlm.evaluate import Evaluator


# -----------------------
# Utility (coerenti con train)
# -----------------------

def is_classification(task: str) -> bool:
    """Ritorna True se il task è di classificazione, False se regressione."""
    return task.lower() in {"gender", "ethnicity", "emotion", "facial emotion"}

def get_num_classes_for_task(task: str) -> int:
    """Ritorna il numero di classi attese per un dato task di classificazione."""
    t = task.lower()
    if t == "gender":
        return 2
    if t in {"emotion", "facial emotion"}:
        return 7
    if t == "ethnicity":
        return 4
    raise ValueError(f"Task di classificazione non riconosciuto: {task}")

def collate_keep_pil(batch):
    """
    Collate che conserva le immagini come PIL.Image senza stackarle.
    Restituisce:
      - images_list: lista di PIL.Image
      - targets_list: lista di dict target originali
    """
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list

# -----------------------
# Argparse
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probe testing (PIL → processor)")
    # Model args
    parser.add_argument("--model_name", type=str, default="llava",
                        choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp32",
                        choices=["4bit", "8bit", "fp16", "fp32"])
    # Dataset args (test)
    parser.add_argument("--dataset_name", type=str, default="VggFace2-Test",
                        choices=DatasetFactory.get_available_datasets())
    parser.add_argument("--task", type=str, default="age",
                        help="gender | ethnicity | emotion | age")
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    # Checkpoint
    parser.add_argument("--ckpt_path", type=str, default="linear_probing/checkpoints/llava_fp32_VggFace2-Train_age.pth",
                        help="Checkpoint .pth da cui prendere il modello salvato")

    # Output valutazione (relativo a project_root/evaluate.py)
    parser.add_argument("--eval_subdir", type=str, default=None,
                        help="Sottocartella (relativa a project_root) dove salvare i risultati dell'eval."
                             " Se None: usa 'eval/{model}_{quant}_{dataset}_{task}'")
    return parser.parse_args()


# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Checkpoint default
    if args.ckpt_path is not None:
        if project_root:
            ckpt_dir = Path(project_root) / args.ckpt_path
        else:
            ckpt_dir = Path(args.ckpt_path)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_dir}")
    else:
        raise FileNotFoundError(f"Checkpoint non specificato")

    # Output dir per Evaluator (relativa a project_root)
    if args.eval_subdir is None:
        eval_subdir = Path(project_root) / "eval" / f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}"
    else:
        eval_subdir = Path(args.eval_subdir)

    eval_subdir.mkdir(parents=True, exist_ok=True)

    # Modello e backbone coerenti col training
    model_id = None
    vlm = VLMModelFactory.create_model(args.model_name, model_id, device, args.quantization)
    backbone = vlm.get_vision_backbone()

    # Output head size
    task_lower = args.task.lower()
    is_cls = is_classification(task_lower)
    if is_cls:
        n_out = get_num_classes_for_task(task_lower)
    else:
        n_out = 1  # regressione age

    probe = LinearProbe(backbone=backbone, n_out_classes=n_out, freeze_backbone=True).to(device)
    # Caricamento pesi dal checkpoint salvato dal training
    ckpt = torch.load(ckpt_dir, map_location="cpu", weights_only=True)
    state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    probe.load_state_dict(state_dict, strict=True)
    probe.eval()

    # Dataset di test e DataLoader (PIL → processor)
    test_dataset = DatasetFactory.create_dataset(
        args.dataset_name,
        base_path=args.base_path,
        train=False,
        transform=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_keep_pil
    )

    # Inference loop: produce preds/gts in formato compatibile con Evaluator
    preds, gts = [], []
    with torch.no_grad():
        for images_list, targets_list in tqdm(test_loader, desc="Testing", unit="batch"):
            # Passa direttamente la lista di PIL al probe; il processor farà il resto
            logits = probe(images_list)

            if is_cls:
                pred_idx = logits.argmax(dim=1).cpu().tolist()
                for i, tgt in enumerate(targets_list):
                    p = {}
                    g = {}
                    key = "emotion" if task_lower.startswith("facial") or task_lower == "emotion" else args.task
                    p[key] = int(pred_idx[i])
                    g[key] = int(tgt.get(key, -1))
                    preds.append(p)
                    gts.append(g)
            else:
                # Regressione age: output [B,1]
                pred_age = logits.squeeze(1).cpu().tolist()
                for i, tgt in enumerate(targets_list):
                    p = {"age": float(pred_age[i])}
                    g = {"age": float(tgt.get("age", -1))}
                    preds.append(p)
                    gts.append(g)

    # Valutazione e salvataggio risultati
    output_dir_rel = eval_subdir
    Evaluator.evaluate(preds, gts, output_dir=output_dir_rel, dataset_name=args.dataset_name)
    print(f"Valutazione completata. Risultati salvati in datasets_vlm/{output_dir_rel}")


if __name__ == "__main__":
    main()