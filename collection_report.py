import argparse
import json
import os
import sys
import textwrap
import traceback
from typing import Dict, List, Sequence

import yaml
from mpi4py import MPI
from openai import OpenAI


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping.")
    return config


def validate_config(cfg: Dict) -> Dict:
    report_cfg = cfg.setdefault("collection_reports", {})
    if not report_cfg.get("enabled", False):
        raise ValueError(
            "Collection reports are disabled in the configuration. "
            "Set `collection_reports.enabled` to true.",
        )

    summaries_dir = report_cfg.get("summaries_dir", "summaries")
    report_cfg["summaries_dir"] = expand_path(summaries_dir)

    tags_dir = report_cfg.get("tags_dir")
    if tags_dir:
        report_cfg["tags_dir"] = expand_path(tags_dir)

    report_cfg["output_path"] = expand_path(report_cfg.get("output_path", "collection_report.md"))

    return cfg


def create_openai_client(report_cfg: Dict) -> OpenAI:
    api_key_env = report_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            "Set it to your OpenAI API key before running collection reports.",
        )
    return OpenAI(api_key=api_key)


def read_text_files(directory: str, limit: int) -> Dict[str, str]:
    if not os.path.exists(directory):
        return {}

    items = {}
    filenames = sorted(os.listdir(directory))[:limit] if limit else sorted(os.listdir(directory))
    for fname in filenames:
        path = os.path.join(directory, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            items[os.path.splitext(fname)[0]] = handle.read().strip()
    return items


def combine_data(summaries: Dict[str, str], tags: Dict[str, str], max_chars: int = 15000) -> str:
    lines: List[str] = []
    for item_id, summary in summaries.items():
        lines.append(f"ID: {item_id}")
        lines.append("Summary:")
        lines.append(summary)
        if item_id in tags:
            try:
                tag_data = json.loads(tags[item_id])
                lines.append("Tags:")
                lines.append(json.dumps(tag_data, ensure_ascii=False))
            except json.JSONDecodeError:
                lines.append("Tags:")
                lines.append(tags[item_id])
        lines.append("")
    data = "\n".join(lines)
    if len(data) > max_chars:
        data = data[:max_chars] + "\n\n[Data truncated for report generation.]"
    return data


def generate_report(client: OpenAI, collection_data: str, report_cfg: Dict) -> str:
    prompt_template = report_cfg.get(
        "prompt_template",
        "Produce a research briefing summarizing the following data:\n{collection_data}",
    )
    user_prompt = prompt_template.format(collection_data=collection_data)
    system_prompt = report_cfg.get(
        "system_prompt",
        "You write analytical yet neutral summaries of archival audiovisual collections.",
    )

    response = client.responses.create(
        model=report_cfg.get("model", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=report_cfg.get("temperature", 0.2),
        max_output_tokens=2000,
    )
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a collection-level synthesis report from summaries and tags.",
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

    report_cfg = config["collection_reports"]

    summaries = read_text_files(report_cfg["summaries_dir"], report_cfg.get("max_items", 200))
    if not summaries:
        sys.exit(f"No summaries found in {report_cfg['summaries_dir']}.")

    tags_dir = report_cfg.get("tags_dir")
    tag_data: Dict[str, str] = {}
    if tags_dir:
        tag_data = read_text_files(tags_dir, report_cfg.get("max_items", 200))

    collection_data = combine_data(summaries, tag_data)

    client = create_openai_client(report_cfg)
    report_text = generate_report(client, collection_data, report_cfg)

    os.makedirs(os.path.dirname(report_cfg["output_path"]), exist_ok=True)
    with open(report_cfg["output_path"], "w", encoding="utf-8") as handle:
        handle.write(report_text + "\n")

    print(f"Saved collection report to {report_cfg['output_path']}")


if __name__ == "__main__":
    main()

