import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
project_root = os.getenv("PYTHONPATH")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from probing.train.utils import load_config
from probing.train.singletask_trainer import SingleTaskTrainer
from probing.train.multitask_trainer import MultiTaskTrainer

def deep_merge(base: dict, override: dict) -> dict:
    """Merge ricorsivo: i valori in override sostituiscono/estendono base."""
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def build_cfg_from_profile(yaml_cfg: dict, profile: str, cfg_path: Path) -> dict:
    if profile not in ("single", "multi"):
        raise ValueError("profile deve essere 'single' o 'multi'")
    common = yaml_cfg.get("common", {})
    branch = yaml_cfg.get(profile, {})
    cfg = deep_merge(common, branch)
    # vincoli minimi
    if profile == "single":
        if "task" not in cfg:
            raise ValueError("Sezione 'single' deve definire 'task'")
    else:
        if "tasks" not in cfg or not cfg["tasks"]:
            raise ValueError("Sezione 'multi' deve definire 'tasks' (lista)")
        cfg["tasks"] = [str(t).lower() for t in cfg["tasks"]]
    cfg["_cfg_path"] = str(cfg_path)
    return cfg

def make_run_name(cfg: dict, trainer_name: str) -> str:
    m = cfg["model"]
    model_name   = m["name"]
    quantization = m.get("quantization")
    deeper = bool(m.get("deeper_head", False))
    head_tag = "deeper" if deeper else "linear"

    if trainer_name == "multi":
        tasks = [t.lower() for t in cfg["tasks"]]
        uw_cfg = (cfg["train"].get("uncertainty_weighting") or {})
        uw_flag = "_uw" if bool(uw_cfg.get("enabled", False)) else ""
        return f"{model_name}_{quantization}_{'-'.join(tasks)}_{head_tag}{uw_flag}"
    else:
        task = str(cfg.get("task", "task")).lower()
        return f"{model_name}_{quantization}_{task}_{head_tag}"

def main():
    ap = argparse.ArgumentParser(description="Unified training entrypoint (single/multi via profilo)")
    ap.add_argument("--config", type=str, default="configs/train_probe.yaml")
    ap.add_argument("--profile", type=str, choices=["single", "multi"],
                    help="Override del profilo definito nel YAML (single|multi)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    raw = load_config(cfg_path)

    # usa override da CLI se presente, altrimenti il profilo nel YAML
    profile = (args.profile or str(raw.get("profile", ""))).lower()
    if profile not in ("single", "multi"):
        raise ValueError("Specifica il profilo: --profile single|multi oppure profile: single|multi nel YAML")

    cfg = build_cfg_from_profile(raw, profile, cfg_path)
    trainer_name = profile
    run_name = make_run_name(cfg, trainer_name)

    # checkpoint roots originali
    if trainer_name == "multi":
        ckpt_root = Path(project_root or ".") / "probing" / "multitask_probing" / "checkpoints"
        trainer = MultiTaskTrainer(cfg, run_name, ckpt_root)
    else:
        ckpt_root = Path(project_root or ".") / "probing" / "linear_probing" / "checkpoints"
        trainer = SingleTaskTrainer(cfg, run_name, ckpt_root)

    trainer.fit()

if __name__ == "__main__":
    main()