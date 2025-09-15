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

# -------------------------------------------------
# Dataset → Task mapping
# -------------------------------------------------
DATASETS_BY_TASK = {
    "gender":    ["CelebA_HQ", "FairFace", "LFW", "UTKFace", "MiviaGender", "RAF-DB", "VggFace2-Test"],
    "ethnicity": ["FairFace", "UTKFace", "VggFace2-Test"],
    "emotion":   ["RAF-DB"],
    "age":       ["UTKFace", "VggFace2-Test", "FairFace"],
}

# -----------------------
# Utility coerenti col train (head-only)
# -----------------------
def get_num_classes_for_task(task: str) -> int:
    t = task.lower()
    if t == "gender": return 2
    if t == "emotion": return 7
    if t == "ethnicity": return 4
    if t == "age": return 9  # 9 classi: ["0-2","3-9","10-19","20-29","30-39","40-49","50-59","60-69","70+"]
    raise ValueError(f"Task di classificazione non riconosciuto: {task}")

def collate_keep_pil(batch):
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list

def run_eval_one_dataset(args, backbone, device, dataset_name, ckpt_dir_base):
    # Output dir per Evaluator
    base = Path(project_root) if project_root else Path(".")
    eval_dir = base / "linear_probing" / "eval" / f"{args.model_name}_{args.quantization}" / f"{args.task.lower()}" / f"{dataset_name}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Head (sempre classificazione)
    n_out = get_num_classes_for_task(args.task)
    probe = LinearProbe(backbone=backbone, n_out_classes=n_out, freeze_backbone=True).to(device)

    # Caricamento pesi head-only
    ckpt_dir = Path(ckpt_dir_base if not project_root else os.path.join(project_root, ckpt_dir_base)).resolve()
    cls_path = ckpt_dir / "classifier.pt"
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir non trovata: {ckpt_dir}")
    if not cls_path.exists():
        raise FileNotFoundError(f"classifier.pt non trovato in {ckpt_dir}")
    state = torch.load(cls_path, map_location="cpu", weights_only=True)
    probe.classifier.load_state_dict(state, strict=True)
    probe.eval()

    # Dataset di test
    test_dataset = DatasetFactory.create_dataset(dataset_name, base_path=args.base_path, train=False, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, collate_fn=collate_keep_pil)

    # Inference (sempre classificazione)
    preds, gts = [], []
    key = args.task.lower()  # "gender" | "ethnicity" | "emotion" | "age"
    use_amp = (device.type == "cuda")
    with torch.inference_mode():
        for images_list, targets_list in tqdm(test_loader, desc=f"Testing {dataset_name}", unit="batch"):
            with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                logits = probe(images=images_list)
            pred_idx = logits.argmax(dim=1).cpu().tolist()

            for i, tgt in enumerate(targets_list):
                preds.append({key: int(pred_idx[i])})
                gts.append({key: int(tgt.get(key, -1))})

    # Eval
    Evaluator.evaluate(preds, gts, output_dir=eval_dir, dataset_name=dataset_name, age_mode="classification")
    print(f"[OK] {dataset_name}: risultati salvati in {eval_dir}")

# -----------------------
# Argparse
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probe testing (PIL → processor) - HEAD ONLY")
    # Model args
    parser.add_argument("--model_name", type=str, default="llava",
                        choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp32",
                        choices=["4bit", "8bit", "fp16", "fp32"])
    # Dataset args (test)
    parser.add_argument("--dataset_name", type=str, default="auto",
                        help="Nome dataset singolo oppure 'auto' per testare tutti i dataset del task")
    parser.add_argument("--task", type=str, default="emotion",
                        help="gender | ethnicity | emotion | age")
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # Checkpoint directory (HEAD ONLY): contiene classifier.pt
    parser.add_argument("--ckpt_dir", type=str, default="linear_probing/checkpoints/llava_fp32_auto_emotion_head",
                        help="Directory della head (deve contenere classifier.pt)")
    return parser.parse_args()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    task = args.task.lower()
    if task not in DATASETS_BY_TASK:
        raise ValueError(f"Task '{args.task}' non supportato. Scegli tra: {list(DATASETS_BY_TASK.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Modello/backbone
    model_id = None
    vlm = VLMModelFactory.create_model(args.model_name, model_id, device, args.quantization)
    backbone = vlm.get_vision_backbone()
    del vlm

    if args.dataset_name.lower() == "auto":
        datasets = DATASETS_BY_TASK[task]
        if not datasets:
            raise ValueError(f"Nessun dataset configurato per il task '{task}'.")
        print(f"[AUTO] Test su tutti i dataset per '{task}': {datasets}")
        for ds in datasets:
            run_eval_one_dataset(args, backbone, device, ds, args.ckpt_dir)
    else:
        if args.dataset_name not in DATASETS_BY_TASK[task]:
            raise ValueError(f"'{args.dataset_name}' non supporta il task '{task}'. "
                             f"Consentiti: {DATASETS_BY_TASK[task]} o 'auto'.")
        run_eval_one_dataset(args, backbone, device, args.dataset_name, args.ckpt_dir)

if __name__ == "__main__":
    main()