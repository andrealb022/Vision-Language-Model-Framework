import os, sys
from pathlib import Path
import argparse
import yaml  # pip install pyyaml
from dotenv import load_dotenv
load_dotenv()
project_root = os.getenv("PYTHONPATH")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.model_factory import VLMModelFactory
from linear_probing.linear_probe import LinearProbe
from datasets_vlm.dataset_factory import DatasetFactory
from datasets_vlm.evaluate_dataset import Evaluator

# -----------------------
# Utils
# -----------------------
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_test_config(cfg_path: str) -> dict:
    cfg_path = os.path.join(project_root, cfg_path) if project_root else cfg_path
    return load_yaml(cfg_path)

def load_head_config(ckpt_dir: Path) -> dict:
    """
    Carica la config salvata durante il train.
    File di configurazione: head_config.yaml
    """
    p = ckpt_dir / "head_config.yaml"
    if p.exists():
        return load_yaml(p)
    raise FileNotFoundError(f"Nessun file di config trovato in {ckpt_dir} "
                            "(attesi: head_config.yaml / used_config.yaml / train_config.yaml)")

def get_num_classes_for_task(task: str) -> int:
    t = task.lower()
    if t == "gender": return 2
    if t == "emotion": return 7
    if t == "ethnicity": return 4
    if t == "age": return 9
    raise ValueError(f"Task non riconosciuto: {task}")

def collate_keep_pil(batch):
    images_list = [b[0] for b in batch]
    targets_list = [b[1] for b in batch]
    return images_list, targets_list

# -----------------------
# Valutazione su 1 dataset
# -----------------------
def run_eval_one_dataset(
    model_name: str,
    quantization: str,
    task: str,
    deeper_head: bool,
    freeze_backbone: bool,
    dropout_p: float,
    hidden_dim: int,
    backbone,
    device,
    dataset_name: str,
    base_path,
    batch_size: int,
    num_workers: int,
    ckpt_dir: Path,
):
    # Eval dir (default da codice, coerente con train)
    head_type = "deeper" if deeper_head else "linear"
    eval_dir = os.path.join(project_root, "linear_probing", "eval", f"{model_name}_{quantization}_{head_type}", task, dataset_name)
    os.makedirs(eval_dir, exist_ok=True)

    # Head (classificazione)
    n_out = get_num_classes_for_task(task)
    probe = LinearProbe(
        backbone=backbone,
        n_out_classes=n_out,
        freeze_backbone=freeze_backbone,
        deeper_head=deeper_head,
        dropout_p=dropout_p,
        hidden_dim=hidden_dim,
    ).to(device)

    # Caricamento pesi head-only
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint dir non trovata: {ckpt_dir}")
    cls_path = ckpt_dir / "classifier.pt"
    if not cls_path.exists():
        raise FileNotFoundError(f"classifier.pt non trovato in {ckpt_dir}")
    state = torch.load(cls_path, map_location="cpu", weights_only=True)
    probe.classifier.load_state_dict(state, strict=True)
    probe.eval()

    # Dataset di test
    test_dataset = DatasetFactory.create_dataset(dataset_name, base_path=base_path, train=False, transform=None)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_keep_pil
    )

    # Inference -> preds/gts
    preds, gts = [], []
    key = task  # "gender" | "ethnicity" | "emotion" | "age"
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
# Main
# -----------------------
def main():
    # args: solo --config
    ap = argparse.ArgumentParser(description="Linear Probe testing (YAML config) - HEAD ONLY")
    ap.add_argument("--config", type=str, default="linear_probing/config/test_config.yaml", help="Path al file YAML di test")
    args = ap.parse_args()

    # Carica config di test minimale
    tcfg = load_test_config(args.config)
    dcfg = tcfg["data"]
    ecfg = tcfg["eval"]

    base_path   = dcfg.get("base_path", None)
    batch_size  = int(dcfg.get("batch_size", 128))
    num_workers = int(dcfg.get("num_workers", 8))

    # Risolvi ckpt dir
    ckpt_from = ecfg.get("ckpt_from")
    if not ckpt_from:
        raise ValueError("Nel test YAML devi specificare eval.ckpt_from (dir contenente classifier.pt e head_config.yaml).")
    ckpt_dir = Path(project_root, ckpt_from) if (project_root and not os.path.isabs(ckpt_from)) else Path(ckpt_from)
    ckpt_dir = ckpt_dir.resolve()

    # Carica la head config salvata in train
    hcfg = load_head_config(ckpt_dir)

    # Estrai i parametri del modello/head e il task dalla head config
    # Supporta sia struttura a sezioni (model/task/train/...) sia meta piatto
    if "model" in hcfg:
        mcfg = hcfg["model"]
        model_name   = mcfg["name"]
        quantization = mcfg["quantization"]
        deeper_head  = bool(mcfg.get("deeper_head", False))
        freeze_bb    = bool(mcfg.get("freeze_backbone", True))
        dropout_p    = float(mcfg.get("dropout_p", 0.3))
        hidden_dim   = int(mcfg.get("hidden_dim", 512))
    else:
        # fallback: meta salvato piatto
        model_name   = hcfg.get("model_name")
        quantization = hcfg.get("quantization", "fp32")
        deeper_head  = bool(hcfg.get("deeper_head", False))
        freeze_bb    = bool(hcfg.get("freeze_backbone", True))
        dropout_p    = float(hcfg.get("dropout_p", 0.3))
        hidden_dim   = int(hcfg.get("hidden_dim", 512))
    task = hcfg.get("task").lower()

    # Device + backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    vlm = VLMModelFactory.create_model(model_name, model_id=None, device=device, quantization=quantization)
    backbone = vlm.get_vision_backbone()
    del vlm

    # Costruisci la lista di dataset da testare
    eval_dataset_name = (ecfg.get("dataset_name", "auto") or "auto")
    if eval_dataset_name.lower() == "auto":
        if not hasattr(DatasetFactory, "TASK_TO_DATASETS"):
            raise RuntimeError("DatasetFactory non espone TASK_TO_DATASETS.")
        if task not in DatasetFactory.TASK_TO_DATASETS:
            raise ValueError(f"Task '{task}' non supportato dalla factory.")
        datasets_to_test = DatasetFactory.TASK_TO_DATASETS[task]
        print(f"[AUTO] Test su tutti i dataset per '{task}': {datasets_to_test}")
    else:
        datasets_to_test = [eval_dataset_name]
        if hasattr(DatasetFactory, "TASK_TO_DATASETS"):
            allowed = DatasetFactory.TASK_TO_DATASETS.get(task, [])
            if eval_dataset_name not in allowed:
                print(f"[WARN] '{eval_dataset_name}' non è nella lista factory per '{task}': {allowed}")

    # Esegui eval per ogni dataset
    for ds in datasets_to_test:
        run_eval_one_dataset(
            model_name=model_name,
            quantization=quantization,
            task=task,
            deeper_head=deeper_head,
            freeze_backbone=freeze_bb,
            dropout_p=dropout_p,
            hidden_dim=hidden_dim,
            backbone=backbone,
            device=device,
            dataset_name=ds,
            base_path=base_path,
            batch_size=batch_size,
            num_workers=num_workers,
            ckpt_dir=ckpt_dir,
        )

if __name__ == "__main__":
    main()