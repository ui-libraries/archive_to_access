import argparse
import json
import os
import statistics
import sys
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

import yaml


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping.")
    return config


def validate_config(cfg: Dict) -> Dict:
    qc_cfg = cfg.setdefault("quality_control", {})
    if not qc_cfg.get("enabled", False):
        raise ValueError(
            "Quality control is disabled in the configuration. "
            "Set `quality_control.enabled` to true.",
        )

    summary_dir = qc_cfg.get("summary_dir", "summaries")
    qc_cfg["summary_dir"] = expand_path(summary_dir)

    transcript_dir = qc_cfg.get("transcript_dir")
    if transcript_dir:
        qc_cfg["transcript_dir"] = expand_path(transcript_dir)
    else:
        outputs_cfg = cfg.get("outputs", {})
        format_dirs = outputs_cfg.get("format_dirs", {})
        transcript_format = qc_cfg.get("transcript_format", "txt")
        qc_cfg["transcript_dir"] = expand_path(format_dirs.get(transcript_format, "")) if format_dirs else None

    if qc_cfg.get("frame_descriptions_dirs"):
        qc_cfg["frame_descriptions_dirs"] = [
            expand_path(path) for path in qc_cfg["frame_descriptions_dirs"]
        ]
    else:
        frame_cfg = cfg.get("frame_descriptions", {})
        if frame_cfg.get("enabled"):
            qc_cfg["frame_descriptions_dirs"] = [
                expand_path(frame_cfg.get("output_dir", "frame_descriptions")),
            ]
        else:
            qc_cfg["frame_descriptions_dirs"] = []

    qc_cfg["output_csv"] = expand_path(qc_cfg.get("output_csv", "quality_metrics.csv"))

    return cfg


def read_text(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def list_files(directory: str, extension: str) -> List[str]:
    if not directory or not os.path.exists(directory):
        return []
    return sorted(
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(extension)
    )


def compute_summary_metrics(text: str) -> Dict[str, float]:
    words = text.split()
    return {
        "summary_word_count": len(words),
        "summary_char_count": len(text),
    }


def compute_transcript_metrics(text: Optional[str]) -> Dict[str, float]:
    if not text:
        return {"transcript_word_count": 0, "transcript_char_count": 0}
    words = text.split()
    return {
        "transcript_word_count": len(words),
        "transcript_char_count": len(text),
    }


def compute_frame_metrics(item_id: str, cfg: Dict) -> Dict[str, float]:
    directories = cfg.get("frame_descriptions_dirs", [])
    extension = cfg.get("frame_description_extension", "txt").lstrip(".")
    total_frames = 0
    avg_description_length = 0.0
    lengths: List[int] = []
    for directory in directories:
        if not os.path.exists(directory):
            continue
        pattern = f"{item_id}_t"
        for fname in os.listdir(directory):
            if fname.startswith(pattern) and fname.endswith(f".{extension}"):
                total_frames += 1
                with open(os.path.join(directory, fname), "r", encoding="utf-8") as handle:
                    lengths.append(len(handle.read().split()))
    if lengths:
        avg_description_length = statistics.mean(lengths)
    return {
        "frame_description_count": total_frames,
        "avg_frame_description_words": avg_description_length,
    }


def calculate_flags(metrics: Dict[str, float], cfg: Dict) -> Dict[str, bool]:
    min_words = cfg.get("min_summary_words", 30)
    max_words = cfg.get("max_summary_words", 70)
    summary_words = metrics.get("summary_word_count", 0)
    flags = {
        "summary_too_short": summary_words < min_words,
        "summary_too_long": summary_words > max_words,
        "missing_frame_descriptions": metrics.get("frame_description_count", 0) == 0,
        "missing_transcript": metrics.get("transcript_word_count", 0) == 0,
    }
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute quality-control metrics for generated transcripts, summaries, and frame descriptions.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: ./config.yaml).",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        config = validate_config(config)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(f"Configuration error: {exc}")

    qc_cfg = config["quality_control"]
    summaries = list_files(qc_cfg["summary_dir"], ".txt")
    if not summaries:
        sys.exit(f"No summaries found in {qc_cfg['summary_dir']}.")

    transcript_dir = qc_cfg.get("transcript_dir")
    results: List[Dict[str, object]] = []
    for fname in summaries:
        item_id = os.path.splitext(fname)[0]
        summary_text = read_text(os.path.join(qc_cfg["summary_dir"], fname)) or ""
        metrics = compute_summary_metrics(summary_text)

        transcript_text = None
        if transcript_dir:
            transcript_text = read_text(os.path.join(transcript_dir, f"{item_id}.{qc_cfg.get('transcript_format', 'txt')}"))
        metrics.update(compute_transcript_metrics(transcript_text))
        metrics.update(compute_frame_metrics(item_id, qc_cfg))

        flags = calculate_flags(metrics, qc_cfg)
        record = {
            "id": item_id,
            **metrics,
            **{f"flag_{k}": v for k, v in flags.items()},
        }
        results.append(record)

    os.makedirs(os.path.dirname(qc_cfg["output_csv"]), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(qc_cfg["output_csv"], index=False)

    print(f"Wrote quality metrics to {qc_cfg['output_csv']}")


if __name__ == "__main__":
    main()

