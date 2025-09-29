import os
from pathlib import Path
from typing import List
import torch
from probing.train.utils import get_num_classes_for_task
from models.model_factory import VLMModelFactory
from datasets_vlm.dataset_factory import DatasetFactory
from probing.linear_probing.linear_probe import LinearProbe
from probing.test.base_tester import BaseTester

class SingleTaskTester(BaseTester):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.ckpt_from = Path(cfg["eval"]["ckpt_from"])
        if os.getenv("PYTHONPATH") and not self.ckpt_from.is_absolute():
            self.ckpt_from = Path(os.getenv("PYTHONPATH")) / self.ckpt_from
        self.ckpt_from = self.ckpt_from.resolve()
        self.head_cfg = self._load_head_config(self.ckpt_from)

        # estrai meta
        if "model" in self.head_cfg:
            m  = self.head_cfg["model"]
            bb = (m.get("backbone") or {})
            self.model_name   = m["name"]
            self.quantization = m.get("quantization", "fp32")
            self.deeper_head  = bool(m.get("deeper_head", False))
            # preferisci il nuovo campo, poi fallback al vecchio
            self.freeze_bb    = bool(bb.get("freeze", m.get("freeze_backbone", True)))
            self.dropout_p    = float(m.get("dropout_p", 0.3))
            self.hidden_dim   = int(m.get("hidden_dim", 512))
        else:
            # vecchio formato piatto (unchanged)
            self.model_name   = self.head_cfg.get("model_name")
            self.quantization = self.head_cfg.get("quantization", "fp32")
            self.deeper_head  = bool(self.head_cfg.get("deeper_head", False))
            self.freeze_bb    = bool(self.head_cfg.get("freeze_backbone", True))
            self.dropout_p    = float(self.head_cfg.get("dropout_p", 0.3))
            self.hidden_dim   = int(self.head_cfg.get("hidden_dim", 512))
        self.task = str(self.head_cfg.get("task")).lower()

    def _load_head_config(self, ckpt_dir: Path) -> dict:
        for fname in ("head_config.yaml", "run_config.yaml"):  # <- fallback
            p = ckpt_dir / fname
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    import yaml
                    return yaml.safe_load(f)
        raise FileNotFoundError(f"config non trovato in {ckpt_dir}")

    # --- BaseTester impl ---
    def load_backbone(self):
        vlm = VLMModelFactory.create_model(self.model_name, model_id=None, device=self.device, quantization=self.quantization)
        bb = vlm.get_vision_backbone(); del vlm
        return bb

    def load_ckpt_and_build_model(self, backbone):
        n_out = get_num_classes_for_task(self.task)
        probe = LinearProbe(
            backbone=backbone,
            n_out_classes=n_out,
            freeze_backbone=self.freeze_bb,
            deeper_head=self.deeper_head,
            dropout_p=self.dropout_p,
            hidden_dim=self.hidden_dim,
        ).to(self.device).eval()

        # Supporta sia classifier.pt (head-only) sia model.pt (full)
        cls_path = self.ckpt_from / "classifier.pt"
        model_path = self.ckpt_from / "model.pt"
        if cls_path.exists():
            state = torch.load(cls_path, map_location="cpu")
            probe.classifier.load_state_dict(state, strict=True)
        elif model_path.exists():
            blob = torch.load(model_path, map_location="cpu")
            if isinstance(blob, dict) and "model" in blob:
                probe.load_state_dict(blob["model"], strict=False)
            else:
                probe.load_state_dict(blob, strict=False)
        else:
            raise FileNotFoundError(f"Nessun checkpoint trovato in {self.ckpt_from} (classifier.pt|model.pt)")
        return probe

    def iter_tasks(self) -> List[str]:
        return [self.task]

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
        logits = model(images=images_list)  # [B, C]
        return logits.argmax(dim=1).cpu().tolist()

    def build_eval_dir(self, task: str, dataset_name: str) -> str:
        head_type = "deeper" if self.deeper_head else "linear"
        base = Path(os.getenv("PYTHONPATH") or ".", "probing", "linear_probing", "eval", f"{self.model_name}_{self.quantization}_{head_type}")
        return str(base / task / dataset_name)

    def dataset_obj(self, dataset_name: str):
        return DatasetFactory.create_dataset(dataset_name, base_path=self.base_path, split="test", transform=None)