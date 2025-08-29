"""
Linear Probe - Test/Evaluation Script (compatibile con:
 - salvataggi LoRA: adapter dir (PEFT) + modules_to_save (classifier)
 - salvataggi no-LoRA: directory con solo classifier.pt
)

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

# PEFT (per la modalità LoRA)
from peft import PeftModel

# -----------------------
# Utility coerenti con train
# -----------------------
def is_classification(task: str) -> bool:
    return task.lower() in {"gender", "ethnicity", "emotion", "facial emotion"}

def get_num_classes_for_task(task: str) -> int:
    t = task.lower()
    if t == "gender": return 2
    if t in {"emotion", "facial emotion"}: return 7
    if t == "ethnicity": return 4
    raise ValueError(f"Task di classificazione non riconosciuto: {task}")

def collate_keep_pil(batch):
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list

# -----------------------
# Argparse
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probe testing (PIL → processor) compatibile con LoRA/head-only")
    # Model args
    parser.add_argument("--model_name", type=str, default="llava",
                        choices=VLMModelFactory.get_available_models())
    parser.add_argument("--quantization", type=str, default="fp32",
                        choices=["4bit", "8bit", "fp16", "fp32"])
    # Dataset args (test)
    parser.add_argument("--dataset_name", type=str, default="VggFace2-Test",
                        choices=DatasetFactory.get_available_datasets())
    parser.add_argument("--task", type=str, default="gender",
                        help="gender | ethnicity | emotion | age")
    parser.add_argument("--base_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # Checkpoint directory (non .pth):
    # - se LoRA: directory adapter PEFT (contiene adapter_config + modules_to_save della head)
    # - se no-LoRA: directory della head (contiene classifier.pt)
    parser.add_argument("--ckpt_dir", type=str, default="linear_probing/checkpoints/llava_fp32_RAF-DB_gender_adapters",
                        help="Directory di checkpoint: adapters (LoRA) oppure head-only (classifier.pt)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Imposta True se il checkpoint è una directory PEFT di adapter LoRA")

    # Output valutazione
    parser.add_argument("--eval_subdir", type=str, default=None,
                        help="Sottocartella (relativa a project_root) per salvare i risultati dell'eval. "
                             "Se None: usa 'eval/{model}_{quant}_{dataset}_{task}'")
    return parser.parse_args()

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Verifica directory di checkpoint
    if args.ckpt_dir is None:
        raise FileNotFoundError("Checkpoint directory non specificata (--ckpt_dir).")
    ckpt_dir = Path(args.ckpt_dir if not project_root else os.path.join(project_root, args.ckpt_dir))
    ckpt_dir = ckpt_dir.resolve()
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Directory checkpoint non trovata: {ckpt_dir}")

    # Output dir per Evaluator (relativa a project_root)
    if args.eval_subdir is None:
        base = Path(project_root) if project_root else Path(".")
        eval_subdir = base / "eval" / f"{args.model_name}_{args.quantization}_{args.dataset_name}_{args.task}"
    else:
        eval_subdir = Path(args.eval_subdir)
        if project_root and not eval_subdir.is_absolute():
            eval_subdir = Path(project_root) / eval_subdir
    eval_subdir.mkdir(parents=True, exist_ok=True)

    # Modello e backbone coerenti col training
    model_id = None
    vlm = VLMModelFactory.create_model(args.model_name, model_id, device, args.quantization)
    backbone = vlm.get_vision_backbone()
    del vlm

    # Output head size
    task_lower = args.task.lower()
    is_cls = is_classification(task_lower)
    n_out = get_num_classes_for_task(task_lower) if is_cls else 1

    # LinearProbe con backbone SEMPRE frozen
    probe = LinearProbe(backbone=backbone, n_out_classes=n_out, freeze_backbone=True).to(device)

    # Caricamento pesi:
    if args.use_lora:
        # Directory PEFT di adapter LoRA + modules_to_save (classifier)
        # (ci aspettiamo file tipo adapter_config.json e adapter_model.bin/safetensors)
        print(f"[LOAD] Modalità LoRA: carico adapter+head da {ckpt_dir}")
        probe = PeftModel.from_pretrained(probe, ckpt_dir.as_posix()).to(device).eval()
    else:
        # Directory head-only con classifier.pt
        cls_path = ckpt_dir / "classifier.pt"
        if not cls_path.exists():
            raise FileNotFoundError(f"classifier.pt non trovato in {ckpt_dir}. "
                                    f"Se il checkpoint è LoRA, aggiungi --use_lora.")
        print(f"[LOAD] Modalità head-only: carico classifier da {cls_path}")
        state = torch.load(cls_path, map_location="cpu")
        probe.classifier.load_state_dict(state, strict=True)
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

    # Inference loop: produce preds/gts per Evaluator
    preds, gts = [], []
    use_amp = (device.type == "cuda")
    probe.eval()

    with torch.inference_mode():
        for images_list, targets_list in tqdm(test_loader, desc="Testing", unit="batch"):
            # PEFT “preferisce” kwargs; passiamo sempre images=...
            with torch.autocast(device_type='cuda' if use_amp else 'cpu', dtype=torch.float16, enabled=use_amp):
                logits = probe(images=images_list)

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
    Evaluator.evaluate(preds, gts, output_dir=eval_subdir, dataset_name=args.dataset_name)
    print(f"Valutazione completata. Risultati salvati in {eval_subdir}")

if __name__ == "__main__":
    main()