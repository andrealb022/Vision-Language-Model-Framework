import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
project_root = os.getenv("PYTHONPATH")
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from probing.train.utils import load_config
from probing.test.singletask_tester import SingleTaskTester
from probing.test.multitask_tester import MultiTaskTester

def deep_merge(base: dict, override: dict) -> dict:
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
    if profile not in ("single","multi"):
        raise ValueError("profile deve essere 'single' o 'multi'")
    common = yaml_cfg.get("common", {})
    branch = yaml_cfg.get(profile, {})
    cfg = deep_merge(common, branch)
    if "eval" not in cfg:
        raise ValueError("La sezione selezionata deve definire 'eval' (ckpt_from, dataset_name)")
    cfg["_cfg_path"] = str(cfg_path)
    return cfg

def main():
    ap = argparse.ArgumentParser(description="Unified testing entrypoint (single/multi via profilo)")
    ap.add_argument("--config", type=str, default="configs/test_probe.yaml",
                    help="YAML con common + profili single/multi")
    ap.add_argument("--profile", type=str, choices=["single","multi"],
                    help="Override del profilo definito nel YAML (single|multi)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    raw = load_config(cfg_path)

    # usa override da CLI se presente, altrimenti il profilo nel YAML
    profile = (args.profile or str(raw.get("profile",""))).lower()
    if profile not in ("single","multi"):
        raise ValueError("Specifica il profilo: --profile single|multi oppure profile: single|multi nel YAML")

    cfg = build_cfg_from_profile(raw, profile, cfg_path)

    tester = MultiTaskTester(cfg) if profile == "multi" else SingleTaskTester(cfg)
    tester.run()

if __name__ == "__main__":
    main()