import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

import yaml


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping.")
    return config


PIPELINE_CMDS = {
    "transcribe": ["parallel_transcription_tool/transcribe_collection.py"],
    "keyframes": ["parallel_transcription_tool/extract_keyframes.py"],
    "describe_frames": ["parallel_transcription_tool/describe_frames.py"],
    "summarize": ["parallel_transcription_tool/summarize_collection.py"],
    "tags": ["parallel_transcription_tool/generate_tags.py"],
    "accessibility": ["parallel_transcription_tool/generate_accessibility.py"],
    "collection_report": ["parallel_transcription_tool/collection_report.py"],
    "preview": ["parallel_transcription_tool/build_preview.py"],
    "quality_metrics": ["parallel_transcription_tool/quality_metrics.py"],
    "iiif": ["parallel_transcription_tool/build_iiif_manifest.py"],
    "catalog_export": ["parallel_transcription_tool/export_catalog.py"],
    "search_index": ["parallel_transcription_tool/build_search_index.py"],
    "clustering": ["parallel_transcription_tool/cluster_visuals.py"],
}


def run_step(step: str, config_path: str, mpirun: bool, ranks: int, provenance: Dict) -> bool:
    cmd = PIPELINE_CMDS.get(step)
    if not cmd:
        print(f"[run_pipeline] Unknown step '{step}', skipping.")
        return False

    if mpirun:
        command = ["mpirun", "-np", str(ranks), "python"] + cmd + ["--config", config_path]
    else:
        command = ["python"] + cmd + ["--config", config_path]

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

