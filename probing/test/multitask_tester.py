import os
from pathlib import Path
from typing import List
import torch
from probing.train.utils import get_num_classes_for_task
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from probing.models.multitask_probe import MultiTaskProbe
from probing.test.base_tester import BaseTester

class MultiTaskTester(BaseTester):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.ckpt_from = Path(cfg["eval"]["ckpt_from"])
        if os.getenv("PYTHONPATH") and not self.ckpt_from.is_absolute():
            self.ckpt_from = Path(os.getenv("PYTHONPATH")) / self.ckpt_from
        self.ckpt_from = self.ckpt_from.resolve()
        self.head_cfg = self._load_head_config(self.ckpt_from)

        # meta
        m = self.head_cfg["model"] if "model" in self.head_cfg else {}
        self.model_name   = m.get("name", self.head_cfg.get("model_name"))
        self.quantization = m.get("quantization", self.head_cfg.get("quantization", "fp32"))
        self.deeper_head  = bool(m.get("deeper_head", self.head_cfg.get("deeper_heads", False)))
        self.freeze_bb    = bool(m.get("freeze_backbone", self.head_cfg.get("freeze_backbone", False)))
        self.dropout_p    = float(m.get("dropout_p", self.head_cfg.get("dropout_p", 0.3)))
        self.hidden_dim   = int(m.get("hidden_dim", self.head_cfg.get("hidden_dim", 512)))

        # tasks dalla config salvata
        if "tasks" in self.head_cfg:
            self.tasks = [t.lower() for t in self.head_cfg["tasks"]]
        elif "train" in self.head_cfg and "tasks" in self.head_cfg["train"]:
            self.tasks = [t.lower() for t in self.head_cfg["train"]["tasks"]]
        else:
            raise ValueError("Impossibile determinare i tasks dal checkpoint config.")

        # run_name (per creare path eval coerenti)
        # Usa la cartella del ckpt come run_name
        self.run_name = self.ckpt_from.name

    def _load_head_config(self, ckpt_dir: Path) -> dict:
        p = ckpt_dir / "head_config.yaml"
        if not p.exists():
            raise FileNotFoundError(f"head_config.yaml non trovato in {ckpt_dir}")
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --- BaseTester impl ---
    def load_backbone(self):
        vlm = VLMModelFactory.create_model(self.model_name, model_id=None, device=self.device, quantization=self.quantization)
        bb = vlm.get_vision_backbone(); del vlm
        return bb

    def load_ckpt_and_build_model(self, backbone):
        tasks_n = {t: get_num_classes_for_task(t) for t in self.tasks}
        probe = MultiTaskProbe(
            backbone=backbone,
            tasks=tasks_n,
            freeze_backbone=self.freeze_bb,
            dropout_p=self.dropout_p,
            deeper_heads=self.deeper_head,
            hidden_dim=self.hidden_dim,
        ).to(self.device).eval()

        # Carica model.pt (full state)
        model_path = self.ckpt_from / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"model.pt non trovato in {self.ckpt_from}")
        blob = torch.load(model_path, map_location="cpu")
        if isinstance(blob, dict) and "model" in blob:
            probe.load_state_dict(blob["model"], strict=False)
        elif isinstance(blob, dict) and "probe" in blob:
            probe.load_state_dict(blob["probe"], strict=False)
        else:
            probe.load_state_dict(blob, strict=False)
        return probe

    def iter_tasks(self) -> List[str]:
        return self.tasks

    def datasets_for_task(self, task: str) -> List[str]:
        ecfg = self.cfg["eval"]
        name = (ecfg.get("dataset_name", "auto") or "auto").lower()
        if name == "auto":
            if not hasattr(DatasetFactory, "TASK_TO_DATASETS_TEST") or task not in DatasetFactory.TASK_TO_DATASETS_TEST:
                raise RuntimeError(f"DatasetFactory.TASK_TO_DATASETS_TEST non disponibile per {task}")
            return DatasetFactory.TASK_TO_DATASETS_TEST[task]
        return [ecfg["dataset_name"]]

    def predict_step(self, model, batch, task: str):
        images_list, _ = batch
        out = model(images_list)  # dict {'logits': {task: [B,C], ...}}
        logits = out["logits"][task]
        return logits.argmax(dim=1).cpu().tolist()

    def build_eval_dir(self, task: str, dataset_name: str) -> str:
        base = Path(os.getenv("PYTHONPATH") or ".", "probing", "multitask_probing", "eval", self.run_name)
        return str(base / task / dataset_name)

    def dataset_obj(self, dataset_name: str):
        return DatasetFactory.create_dataset(dataset_name, base_path=self.base_path, split="test", transform=None)