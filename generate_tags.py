import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np
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
    tagging_cfg = cfg.setdefault("tagging", {})
    if not tagging_cfg.get("enabled", False):
        raise ValueError(
            "Tagging is disabled in the configuration. "
            "Set `tagging.enabled` to true.",
        )

    outputs_cfg = cfg.setdefault("outputs", {})
    base_dir = outputs_cfg.get("base_dir", "transcripts")
    outputs_cfg["base_dir"] = expand_path(base_dir)

    formats: Sequence[str] = outputs_cfg.get("formats", ["txt"])
    if not formats:
        raise ValueError("`outputs.formats` must contain at least one entry.")
    format_dirs = {}
    for fmt in formats:
        format_dirs[fmt] = os.path.join(outputs_cfg["base_dir"], fmt)
    outputs_cfg["format_dirs"] = {fmt: expand_path(path) for fmt, path in format_dirs.items()}

    transcript_format = tagging_cfg.get("transcript_format", "txt")
    transcripts_subdir = tagging_cfg.get("transcripts_subdir")
    if transcripts_subdir:
        tagging_cfg["transcript_dir"] = expand_path(transcripts_subdir)
    else:
        tagging_cfg["transcript_dir"] = outputs_cfg["format_dirs"].get(transcript_format)

    if tagging_cfg["transcript_dir"] is None:
        raise ValueError(
            "Unable to locate transcripts for tagging. "
            "Specify `tagging.transcripts_subdir` or ensure the format is in outputs.formats.",
        )

    tag_output = tagging_cfg.get("output_dir", "tags")
    tagging_cfg["output_dir"] = expand_path(tag_output)

    metadata_csv = tagging_cfg.get("metadata_csv")
    if metadata_csv:
        tagging_cfg["metadata_csv"] = expand_path(metadata_csv)

    return cfg


def ensure_directories(cfg: Dict, rank: int) -> None:
    if rank != 0:
        return
    os.makedirs(cfg["tagging"]["output_dir"], exist_ok=True)


def discover_transcripts(directory: str, extension: str) -> List[str]:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Transcript directory not found: {directory}")
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(f".{extension}")
    ]
    files.sort()
    return files


def load_metadata(cfg: Dict) -> Dict[str, Dict[str, str]]:
    metadata_path = cfg.get("metadata_csv")
    if not metadata_path:
        return {}

    id_column = cfg.get("metadata_id_column", "ID")
    desired_fields: Sequence[str] = cfg.get("metadata_fields", [])

    metadata: Dict[str, Dict[str, str]] = {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if id_column not in reader.fieldnames:
            raise ValueError(
                f"Metadata CSV is missing the id column '{id_column}'. "
                f"Columns available: {reader.fieldnames}",
            )
        missing_fields = [field for field in desired_fields if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(
                f"Metadata CSV missing requested fields: {missing_fields}. "
                f"Columns available: {reader.fieldnames}",
            )
        for row in reader:
            item_id = row[id_column]
            metadata[item_id] = {field: row[field] for field in desired_fields}
    return metadata


def transcript_id_from_path(path: str) -> str:
    filename = os.path.basename(path)
    stem, _ = os.path.splitext(filename)
    return stem


def read_transcript(path: str, max_chars: Optional[int]) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if max_chars and len(text) > max_chars:
        truncation_notice = "\n\n[Transcript truncated for tagging.]"
        text = text[: max_chars - len(truncation_notice)] + truncation_notice
    return text.strip()


def construct_metadata_block(metadata: Dict[str, Dict[str, str]], item_id: str, fallback: str) -> str:
    if item_id not in metadata:
        return fallback
    lines = []
    for key, value in metadata[item_id].items():
        if not value:
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"{pretty_key}: {value}")
    return "\n".join(lines) if lines else fallback


def create_openai_client(tagging_cfg: Dict) -> OpenAI:
    api_key_env = tagging_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            "Set it to your OpenAI API key before running tagging.",
        )
    return OpenAI(api_key=api_key)


def tag_transcript(
    client: OpenAI,
    transcript_text: str,
    metadata_block: str,
    tagging_cfg: Dict,
) -> Dict:
    prompt_template = tagging_cfg.get(
        "prompt_template",
        (
            "Identify key named entities, topical keywords, and themes for this transcript.\n"
            "{metadata_block}\nTranscript:\n{transcript}"
        ),
    )
    user_prompt = prompt_template.format(
        transcript=transcript_text,
        metadata_block=metadata_block,
    )
    system_prompt = tagging_cfg.get(
        "system_prompt",
        (
            "You are an archivist producing JSON metadata with fields "
            '{"entities": [{"type": "...", "label": "..."}], "topics": [], "keywords": []}. '
            "Use lowercase type labels and concise keywords."
        ),
    )

    response = client.responses.create(
        model=tagging_cfg.get("model", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_output_tokens=800,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON response:\n{response.output_text}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel GPT tagging of transcripts for entity/topic metadata.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: ./config.yaml).",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        try:
            config = load_config(args.config)
            config = validate_config(config)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            sys.exit(f"Configuration error: {exc}")
    else:
        config = None

    config = comm.bcast(config, root=0)
    tagging_cfg = config["tagging"]
    ensure_directories(config, rank)

    if rank == 0:
        transcript_dir = tagging_cfg["transcript_dir"]
        transcript_format = tagging_cfg.get("transcript_format", "txt")
        transcript_files = discover_transcripts(transcript_dir, transcript_format)
        if not transcript_files:
            sys.exit(f"No transcripts found in {transcript_dir} with extension '.{transcript_format}'.")
        max_chars = tagging_cfg.get("max_transcript_chars")
        metadata = load_metadata(tagging_cfg)
    else:
        transcript_files = None
        max_chars = None
        metadata = None

    transcript_files = comm.bcast(transcript_files, root=0)
    max_chars = comm.bcast(max_chars, root=0)
    metadata = comm.bcast(metadata, root=0)

    files_for_rank = np.array_split(transcript_files, size)[rank]
    files_for_rank = list(files_for_rank)

    if rank == 0:
        print(
            f"Starting tagging with {size} MPI ranks, "
            f"{len(transcript_files)} transcripts, "
            f"model '{tagging_cfg.get('model', 'gpt-4o-mini')}'.",
            flush=True,
        )

    client = create_openai_client(tagging_cfg)
    output_dir = tagging_cfg["output_dir"]
    overwrite = tagging_cfg.get("overwrite", False)
    metadata_fallback = tagging_cfg.get("metadata_missing_fallback", "Archival metadata unavailable.")

    processed = 0
    started_at = time.time()
    errors: List[Dict[str, str]] = []

    for idx, transcript_path in enumerate(files_for_rank, start=1):
        item_id = transcript_id_from_path(transcript_path)
        output_path = os.path.join(output_dir, f"{item_id}.json")

        if not overwrite and os.path.exists(output_path):
            continue

        try:
            transcript_text = read_transcript(transcript_path, max_chars)
            metadata_block = construct_metadata_block(metadata, item_id, metadata_fallback)
            tags = tag_transcript(client, transcript_text, metadata_block, tagging_cfg)
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(tags, handle, ensure_ascii=False, indent=2)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": transcript_path, "error": str(exc)})
            if tagging_cfg.get("verbose"):
                print(
                    f"[rank {rank}] Error tagging {transcript_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        interval = config.get("logging", {}).get("progress_interval", 5)
        if interval and processed and processed % interval == 0:
            elapsed = time.time() - started_at
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"[rank {rank}] Generated tags for {processed} transcripts "
                f"({idx}/{len(files_for_rank)}) at {rate:.2f} items/sec.",
                flush=True,
            )

    gathered_errors = comm.gather(errors, root=0)
    gathered_counts = comm.gather(processed, root=0)

    if rank == 0:
        total_processed = sum(gathered_counts)
        print(
            f"Tagging complete. Generated tags for {total_processed} transcripts "
            f"across {size} ranks at {datetime.now().isoformat()}",
            flush=True,
        )
        aggregated_errors = [item for sublist in gathered_errors for item in sublist]
        if aggregated_errors:
            print(
                f"{len(aggregated_errors)} transcripts encountered errors. See details below:",
                file=sys.stderr,
            )
            for err in aggregated_errors:
                print(
                    f" - {err['path']}: {err['error']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()

