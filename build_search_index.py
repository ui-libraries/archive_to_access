import argparse
import json
import os
import sqlite3
import sys
import traceback
from typing import Dict, Optional

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
    index_cfg = cfg.setdefault("search_index", {})
    if not index_cfg.get("enabled", False):
        raise ValueError(
            "Search index generation is disabled in the configuration. "
            "Set `search_index.enabled` to true.",
        )

    index_cfg["sqlite_path"] = expand_path(index_cfg.get("sqlite_path", "search_index.db"))
    index_cfg["summaries_dir"] = expand_path(cfg.get("summarization", {}).get("output_dir", "summaries"))
    index_cfg["tags_dir"] = expand_path(cfg.get("tagging", {}).get("output_dir", "tags"))

    transcript_dir = index_cfg.get("transcript_dir")
    if transcript_dir:
        index_cfg["transcript_dir"] = expand_path(transcript_dir)
    else:
        outputs_cfg = cfg.get("outputs", {})
        format_dirs = outputs_cfg.get("format_dirs", {})
        transcript_format = index_cfg.get("transcript_format", "txt")
        index_cfg["transcript_dir"] = expand_path(format_dirs.get(transcript_format, "")) if format_dirs else None

    return cfg


def read_text(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_json(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
            return json.dumps(data, ensure_ascii=False)
        except json.JSONDecodeError:
            return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a SQLite FTS index over transcripts, summaries, and tags.",
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

    index_cfg = config["search_index"]
    summaries_dir = index_cfg["summaries_dir"]
    if not os.path.exists(summaries_dir):
        sys.exit(f"Summaries directory not found: {summaries_dir}")

    os.makedirs(os.path.dirname(index_cfg["sqlite_path"]), exist_ok=True)
    conn = sqlite3.connect(index_cfg["sqlite_path"])
    try:
        conn.execute("DROP TABLE IF EXISTS documents")
        conn.execute(
            "CREATE VIRTUAL TABLE documents USING fts5(id, summary, transcript, tags, tokenize='porter')",
        )
        for fname in sorted(os.listdir(summaries_dir)):
            if not fname.endswith(".txt"):
                continue
            item_id = os.path.splitext(fname)[0]
            summary = read_text(os.path.join(summaries_dir, fname)) if index_cfg.get("include_summaries", True) else ""
            transcript = ""
            if index_cfg.get("include_transcripts", True) and index_cfg.get("transcript_dir"):
                transcript = read_text(
                    os.path.join(
                        index_cfg["transcript_dir"],
                        f"{item_id}.{index_cfg.get('transcript_format', 'txt')}",
                    ),
                )
            tags = ""
            if index_cfg.get("include_tags", True):
                tags = load_json(os.path.join(index_cfg["tags_dir"], f"{item_id}.json"))
            conn.execute(
                "INSERT INTO documents(id, summary, transcript, tags) VALUES (?, ?, ?, ?)",
                (item_id, summary, transcript, tags),
            )
        conn.commit()
    finally:
        conn.close()

    print(f"Created search index at {index_cfg['sqlite_path']}")


if __name__ == "__main__":
    main()

