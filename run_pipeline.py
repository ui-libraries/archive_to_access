import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

import yaml


SCRIPT_ROOT = os.path.dirname(os.path.abspath(__file__))


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping.")
    return config


PIPELINE_CMDS = {
    "transcribe": "transcribe_collection.py",
    "keyframes": "extract_keyframes.py",
    "describe_frames": "describe_frames.py",
    "summarize": "summarize_collection.py",
    "tags": "generate_tags.py",
    "accessibility": "generate_accessibility.py",
    "collection_report": "collection_report.py",
    "preview": "build_preview.py",
    "quality_metrics": "quality_metrics.py",
    "iiif": "build_iiif_manifest.py",
    "catalog_export": "export_catalog.py",
    "search_index": "build_search_index.py",
    "clustering": "cluster_visuals.py",
}


def resolve_script(script: str) -> str:
    return os.path.join(SCRIPT_ROOT, script)


def run_step(step: str, config_path: str, mpirun: bool, ranks: int, provenance: Dict) -> bool:
    script = PIPELINE_CMDS.get(step)
    if not script:
        print(f"[run_pipeline] Unknown step '{step}', skipping.")
        return False

    script_path = resolve_script(script)
    if mpirun:
        command = ["mpirun", "-np", str(ranks), "python", script_path, "--config", config_path]
    else:
        command = ["python", script_path, "--config", config_path]

    print(f"\n[run_pipeline] Running step '{step}': {' '.join(command)}")
    start = time.time()
    process = subprocess.run(command)
    end = time.time()
    provenance_entry = {
        "step": step,
        "command": command,
        "start": datetime.fromtimestamp(start).isoformat(),
        "end": datetime.fromtimestamp(end).isoformat(),
        "duration_sec": end - start,
        "returncode": process.returncode,
    }
    provenance.setdefault("steps", []).append(provenance_entry)
    return process.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run configured pipeline steps in sequence, capturing provenance logs.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: ./config.yaml).",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        help="Override the configured workflow steps (space-separated).",
    )
    parser.add_argument(
        "--no-mpi",
        action="store_true",
        help="Run steps without mpirun even if configured otherwise.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        default=4,
        help="Number of MPI ranks when using mpirun (default: 4).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    workflow_steps: List[str] = []
    if args.steps:
        workflow_steps = args.steps
    else:
        workflow_steps = config.get("workflow", {}).get("steps", [])

    if not workflow_steps:
        print("No workflow steps configured. Specify --steps or set workflow.steps in the config.")
        sys.exit(1)

    provenance_cfg = config.get("provenance", {})
    provenance_enabled = provenance_cfg.get("enabled", True)
    provenance_log = expand_path(provenance_cfg.get("log_path", "provenance_log.jsonl"))

    provenance_record = {
        "config_path": args.config,
        "workflow_steps": workflow_steps,
        "started_at": datetime.utcnow().isoformat(),
        "steps": [],
    }

    success = True
    for step in workflow_steps:
        step_success = run_step(step, args.config, mpirun=not args.no_mpi, ranks=args.ranks, provenance=provenance_record)
        if not step_success:
            print(f"[run_pipeline] Step '{step}' failed with non-zero exit code.")
            success = False
            break

    provenance_record["finished_at"] = datetime.utcnow().isoformat()
    provenance_record["success"] = success

    if provenance_enabled:
        os.makedirs(os.path.dirname(provenance_log), exist_ok=True)
        with open(provenance_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(provenance_record, ensure_ascii=False) + "\n")
        print(f"[run_pipeline] Appended provenance record to {provenance_log}")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

