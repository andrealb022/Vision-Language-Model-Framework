import os, sys, numpy as np, yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from probing.train.utils import collate_keep_pil
from datasets_vlm.evaluate_dataset import Evaluator

class BaseTester:
    """
    Tester base: gestisce device, caricamento backbone, ciclo di inference e
    scrittura risultati. Le sottoclassi implementano:
      - load_ckpt_and_build_model(backbone) -> torch.nn.Module pronto in eval()
      - iter_tasks() -> lista di task da valutare
      - predict_step(model, batch, task) -> (preds: List[int])
      - build_eval_dir(task, dataset_name) -> str/path per l'output
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        dcfg = cfg["data"]
        self.base_path   = dcfg.get("base_path", None)
        self.batch_size  = int(dcfg.get("batch_size", 128))
        self.num_workers = int(dcfg.get("num_workers", 8))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

    # --- API da implementare ---
    def load_backbone(self): raise NotImplementedError
    def load_ckpt_and_build_model(self, backbone): raise NotImplementedError
    def iter_tasks(self) -> List[str]: raise NotImplementedError
    def datasets_for_task(self, task: str) -> List[str]: raise NotImplementedError
    def predict_step(self, model, batch, task: str): raise NotImplementedError
    def build_eval_dir(self, task: str, dataset_name: str) -> str: raise NotImplementedError
    def dataset_obj(self, dataset_name: str):
        raise NotImplementedError  # Factory esterna specifica (dipende dal progetto)

    # --- Inference loop per un (task, dataset) ---
    @torch.no_grad()
    def run_one(self, model, task: str, dataset_name: str):

        ds = self.dataset_obj(dataset_name)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, pin_memory=(self.device.type == "cuda"),
                            collate_fn=collate_keep_pil)

        preds, gts = [], []
        use_amp = (self.device.type == "cuda")
        for images_list, targets_list in loader:
            with torch.autocast(device_type=('cuda' if use_amp else 'cpu'), dtype=torch.float16, enabled=use_amp):
                pred_idxs = self.predict_step(model, (images_list, targets_list), task)

            # target collection
            key = "age" if task == "age" else task
            for i, tgt in enumerate(targets_list):
                preds.append({key: int(pred_idxs[i])})
                gts.append({key: int(tgt.get(key, -1))})

        out_dir = self.build_eval_dir(task, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        Evaluator.evaluate(preds, gts, output_dir=out_dir, dataset_name=dataset_name, age_mode="classification")
        print(f"[OK] {task} @ {dataset_name}: risultati salvati in {out_dir}")

    def run(self):
        # Backbone & model
        backbone = self.load_backbone()
        model = self.load_ckpt_and_build_model(backbone)
        model.eval().to(self.device)

        # Tasks & datasets
        tasks = self.iter_tasks()
        for task in tasks:
            datasets = self.datasets_for_task(task)
            for ds in datasets:
                self.run_one(model, task, ds)
