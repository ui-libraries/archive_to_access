import argparse
import csv
import json
import os
import sys
import traceback
from typing import Dict, List, Optional

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
    export_cfg = cfg.setdefault("catalog_export", {})
    if not export_cfg.get("enabled", False):
        raise ValueError(
            "Catalog export is disabled in the configuration. "
            "Set `catalog_export.enabled` to true.",
        )

    export_cfg["output_csv"] = expand_path(export_cfg.get("output_csv", "catalog_export.csv"))
    export_cfg["summaries_dir"] = expand_path(cfg.get("summarization", {}).get("output_dir", "summaries"))
    export_cfg["tags_dir"] = expand_path(cfg.get("tagging", {}).get("output_dir", "tags"))

    transcript_dir = export_cfg.get("transcript_dir")
    if transcript_dir:
        export_cfg["transcript_dir"] = expand_path(transcript_dir)
    else:
        outputs_cfg = cfg.get("outputs", {})
        format_dirs = outputs_cfg.get("format_dirs", {})
        transcript_format = export_cfg.get("transcript_format", "txt")
        export_cfg["transcript_dir"] = expand_path(format_dirs.get(transcript_format, "")) if format_dirs else None

    metadata_csv = cfg.get("summarization", {}).get("metadata_csv") or cfg.get("tagging", {}).get("metadata_csv")
    if metadata_csv:
        export_cfg["metadata_csv"] = expand_path(metadata_csv)

    return cfg


def read_text(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_json(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metadata(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path or not os.path.exists(path):
        return {}
    metadata: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identifier = row.get("ID") or row.get("id") or row.get("Identifier")
            if not identifier:
                continue
            metadata[identifier] = row
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export enriched catalog metadata combining summaries, tags, and transcripts.",
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

    export_cfg = config["catalog_export"]
    summary_dir = export_cfg["summaries_dir"]
    if not os.path.exists(summary_dir):
        sys.exit(f"Summaries directory not found: {summary_dir}")

    tags_dir = export_cfg["tags_dir"]
    transcript_dir = export_cfg.get("transcript_dir")
    transcript_format = export_cfg.get("transcript_format", "txt")
    metadata = load_metadata(export_cfg.get("metadata_csv"))

    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(summary_dir) if f.endswith(".txt"))
    if not ids:
        sys.exit(f"No summaries found in {summary_dir}.")

    os.makedirs(os.path.dirname(export_cfg["output_csv"]), exist_ok=True)
    fieldnames = [
        "ID",
        "Summary",
        "TranscriptExcerpt",
        "Tags",
    ]
    if metadata:
        extra_fields = sorted({key for row in metadata.values() for key in row.keys()})
        fieldnames.extend(extra_fields)

    with open(export_cfg["output_csv"], "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item_id in ids:
            summary = read_text(os.path.join(summary_dir, f"{item_id}.txt"))
            if not summary:
                continue
            row: Dict[str, str] = {"ID": item_id, "Summary": summary}

            transcript_excerpt = ""
            if export_cfg.get("include_transcript_excerpt", True) and transcript_dir:
                transcript = read_text(os.path.join(transcript_dir, f"{item_id}.{transcript_format}"))
                if transcript:
                    excerpt_len = export_cfg.get("transcript_excerpt_chars", 500)
                    transcript_excerpt = transcript[:excerpt_len]
            row["TranscriptExcerpt"] = transcript_excerpt

            tags = None
            if export_cfg.get("include_tags", True):
                tag_data = load_json(os.path.join(tags_dir, f"{item_id}.json"))
                if tag_data:
                    tags = json.dumps(tag_data, ensure_ascii=False)
            row["Tags"] = tags or ""

            if metadata and item_id in metadata:
                for key, value in metadata[item_id].items():
                    row[key] = value

            writer.writerow(row)

    print(f"Wrote catalog export CSV to {export_cfg['output_csv']}")


if __name__ == "__main__":
    main()

