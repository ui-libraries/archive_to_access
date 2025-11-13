import argparse
import csv
import glob
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

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
    summarization_cfg = cfg.setdefault("summarization", {})
    if not summarization_cfg.get("enabled", False):
        raise ValueError(
            "Summarization is disabled in the configuration. "
            "Set `summarization.enabled` to true.",
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
    outputs_cfg["format_dirs"] = format_dirs

    transcripts_subdir = summarization_cfg.get("transcripts_subdir")
    transcript_format = summarization_cfg.get("transcript_format", "txt")
    format_dirs = outputs_cfg.get("format_dirs")
    transcript_dir = None

    if transcripts_subdir:
        transcript_dir = expand_path(transcripts_subdir)
    elif format_dirs and transcript_format in format_dirs:
        transcript_dir = expand_path(format_dirs[transcript_format])
    else:
        transcript_dir = os.path.join(outputs_cfg["base_dir"], transcript_format)

    summarization_cfg["transcript_dir"] = transcript_dir

    output_dir = summarization_cfg.get("output_dir", "summaries")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(outputs_cfg["base_dir"], "..", output_dir)
    summarization_cfg["output_dir"] = expand_path(output_dir)

    metadata_csv = summarization_cfg.get("metadata_csv")
    if metadata_csv:
        summarization_cfg["metadata_csv"] = expand_path(metadata_csv)

    api_key_env = summarization_cfg.get("api_key_env", "OPENAI_API_KEY")
    summarization_cfg["api_key_env"] = api_key_env

    if summarization_cfg.get("frame_descriptions_dirs"):
        summarization_cfg["frame_descriptions_dirs"] = [
            expand_path(path) for path in summarization_cfg["frame_descriptions_dirs"]
        ]
    else:
        frame_cfg = cfg.get("frame_descriptions", {})
        if frame_cfg.get("enabled"):
            summarization_cfg["frame_descriptions_dirs"] = [
                expand_path(frame_cfg.get("output_dir", "frame_descriptions")),
            ]
        else:
            summarization_cfg["frame_descriptions_dirs"] = []

    summarization_cfg.setdefault("frame_description_extension", "txt")

    return cfg


def ensure_directories(cfg: Dict, rank: int) -> None:
    if rank != 0:
        return
    out_dir = cfg["summarization"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)


def discover_transcripts(transcript_dir: str, extension: str) -> List[str]:
    if not os.path.exists(transcript_dir):
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")
    files = [
        os.path.join(transcript_dir, f)
        for f in os.listdir(transcript_dir)
        if os.path.isfile(os.path.join(transcript_dir, f)) and f.endswith(f".{extension}")
    ]
    files.sort()
    return files


def load_metadata(summarization_cfg: Dict) -> Dict[str, Dict[str, str]]:
    metadata_path = summarization_cfg.get("metadata_csv")
    if not metadata_path:
        return {}

    id_column = summarization_cfg.get("metadata_id_column", "ID")
    desired_fields: Sequence[str] = summarization_cfg.get("metadata_fields", [])

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


def construct_metadata_block(
    metadata: Dict[str, Dict[str, str]],
    item_id: str,
    fallback: str,
) -> str:
    if item_id not in metadata:
        return fallback

    lines = []
    for key, value in metadata[item_id].items():
        if not value:
            continue
        pretty_key = key.replace("_", " ").title()
        lines.append(f"{pretty_key}: {value}")
    return "\n".join(lines) if lines else fallback


def parse_timestamp_from_filename(path: str) -> float:
    stem = os.path.splitext(os.path.basename(path))[0]
    if "_t" not in stem:
        return 0.0
    ts_part = stem.rsplit("_t", 1)[-1]
    try:
        milliseconds = int(ts_part)
    except ValueError:
        milliseconds = 0
    return milliseconds / 1000.0


def build_prompt(
    transcript_text: str,
    metadata_block: str,
    frame_block: str,
    summarization_cfg: Dict,
) -> Dict[str, str]:
    word_limit = summarization_cfg.get("word_limit", 60)
    prompt_template = summarization_cfg.get(
        "prompt_template",
        "Provide a {word_limit}-word summary of the following transcript:\n{transcript}",
    )

    template_args = {
        "word_limit": word_limit,
        "transcript": transcript_text,
        "metadata_block": metadata_block,
        "frame_descriptions": frame_block,
    }
    user_prompt = prompt_template.format(**template_args)

    system_prompt = summarization_cfg.get(
        "system_prompt",
        "You write concise, neutral summaries for archival audiovisual materials.",
    )

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


def create_openai_client(summarization_cfg: Dict) -> OpenAI:
    api_key_env = summarization_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            "Set it to your OpenAI API key before running summarization.",
        )
    return OpenAI(api_key=api_key)


def summarize_transcript(
    client: OpenAI,
    transcript_text: str,
    metadata_block: str,
    frame_block: str,
    summarization_cfg: Dict,
) -> str:
    prompts = build_prompt(transcript_text, metadata_block, frame_block, summarization_cfg)
    model = summarization_cfg.get("model", "gpt-4o-mini")
    temperature = summarization_cfg.get("temperature", 0.2)
    word_limit = summarization_cfg.get("word_limit", 60)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
        ],
        max_output_tokens=word_limit * 4,
        temperature=temperature,
    )
    return response.output_text.strip()


def read_transcript(path: str, max_chars: Optional[int]) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if max_chars and len(text) > max_chars:
        truncation_notice = "\n\n[Transcript truncated for summarization.]"
        text = text[: max_chars - len(truncation_notice)] + truncation_notice
    return text.strip()


def gather_frame_descriptions(
    item_id: str,
    summarization_cfg: Dict,
) -> List[Tuple[float, str]]:
    if not summarization_cfg.get("use_frame_descriptions", False):
        return []

    directories = summarization_cfg.get("frame_descriptions_dirs", [])
    if not directories:
        return []

    extension = summarization_cfg.get("frame_description_extension", "txt").lstrip(".")
    pattern_suffix = f"{item_id}_t*.{extension}"

    entries: List[Tuple[float, str]] = []
    for directory in directories:
        pattern = os.path.join(directory, "**", pattern_suffix)
        for path in glob.glob(pattern, recursive=True):
            timestamp = parse_timestamp_from_filename(path)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    description = handle.read().strip()
            except OSError:
                continue
            entries.append((timestamp, description))

    entries.sort(key=lambda item: item[0])
    limit = summarization_cfg.get("frame_descriptions_limit")
    if limit is not None:
        entries = entries[:limit]
    return entries


def format_frame_descriptions(
    frames: List[Tuple[float, str]],
    summarization_cfg: Dict,
) -> str:
    if not frames:
        return summarization_cfg.get(
            "frame_descriptions_missing_fallback",
            "No frame descriptions available.",
        )

    entry_template = summarization_cfg.get(
        "frame_entry_template",
        "{idx}. ({timestamp:.1f}s) {description}",
    )
    lines = [
        entry_template.format(idx=idx, timestamp=timestamp, description=description)
        for idx, (timestamp, description) in enumerate(frames, start=1)
    ]

    frame_block_template = summarization_cfg.get(
        "frame_block_template",
        "Visual key points:\n{frames}",
    )
    return frame_block_template.format(frames="\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel GPT summarization of transcripts produced by transcribe_collection.py.",
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
    summarization_cfg = config["summarization"]
    ensure_directories(config, rank)

    if rank == 0:
        transcript_dir = summarization_cfg["transcript_dir"]
        transcript_format = summarization_cfg.get("transcript_format", "txt")
        transcript_files = discover_transcripts(transcript_dir, transcript_format)
        if not transcript_files:
            sys.exit(f"No transcripts found in {transcript_dir} with extension '.{transcript_format}'.")
        max_chars = summarization_cfg.get("max_transcript_chars")
        metadata = load_metadata(summarization_cfg)
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
            f"Starting summarization with {size} MPI ranks, "
            f"{len(transcript_files)} transcripts, "
            f"model '{summarization_cfg.get('model', 'gpt-4o-mini')}'.",
            flush=True,
        )

    client = create_openai_client(summarization_cfg)
    output_dir = summarization_cfg["output_dir"]
    overwrite = summarization_cfg.get("overwrite", False)
    metadata_fallback = summarization_cfg.get(
        "metadata_missing_fallback",
        "Archival metadata unavailable.",
    )

    processed = 0
    started_at = time.time()
    errors: List[Dict[str, str]] = []

    for idx, transcript_path in enumerate(files_for_rank, start=1):
        item_id = transcript_id_from_path(transcript_path)
        summary_path = os.path.join(output_dir, f"{item_id}.txt")

        if not overwrite and os.path.exists(summary_path):
            continue

        try:
            transcript_text = read_transcript(transcript_path, max_chars)
            metadata_block = construct_metadata_block(metadata, item_id, metadata_fallback)
            frame_entries = gather_frame_descriptions(item_id, summarization_cfg)
            frame_block = format_frame_descriptions(frame_entries, summarization_cfg)
            summary_text = summarize_transcript(
                client,
                transcript_text,
                metadata_block,
                frame_block,
                summarization_cfg,
            )
            with open(summary_path, "w", encoding="utf-8") as handle:
                handle.write(summary_text)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": transcript_path, "error": str(exc)})
            if summarization_cfg.get("verbose"):
                print(
                    f"[rank {rank}] Error summarizing {transcript_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        interval = config.get("logging", {}).get("progress_interval", 5)
        if interval and processed and processed % interval == 0:
            elapsed = time.time() - started_at
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"[rank {rank}] Completed {processed} summaries "
                f"({idx}/{len(files_for_rank)}) at {rate:.2f} items/sec.",
                flush=True,
            )

    gathered_errors = comm.gather(errors, root=0)
    gathered_counts = comm.gather(processed, root=0)

    if rank == 0:
        total_processed = sum(gathered_counts)
        print(
            f"Summarization complete. Generated {total_processed} summaries "
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

