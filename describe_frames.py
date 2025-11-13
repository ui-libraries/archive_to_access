import argparse
import base64
import csv
import glob
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

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
    fd_cfg = cfg.setdefault("frame_descriptions", {})
    if not fd_cfg.get("enabled", False):
        raise ValueError(
            "Frame descriptions are disabled in the configuration. "
            "Set `frame_descriptions.enabled` to true.",
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

    key_cfg = cfg.get("keyframes", {})
    if not fd_cfg.get("image_extension"):
        fd_cfg["image_extension"] = key_cfg.get("image_extension", "jpg")

    if fd_cfg.get("frames_root"):
        fd_cfg["frames_root"] = expand_path(fd_cfg["frames_root"])
    else:
        if key_cfg.get("enabled"):
            fd_cfg["frames_root"] = expand_path(key_cfg.get("output_dir", "keyframes"))
        else:
            raise ValueError(
                "`frame_descriptions.frames_root` must be set when keyframes are not enabled.",
            )

    fd_cfg["output_dir"] = expand_path(fd_cfg.get("output_dir", "frame_descriptions"))

    transcript_dir = fd_cfg.get("transcripts_subdir")
    transcript_format = fd_cfg.get("transcript_format", "txt")
    if transcript_dir:
        fd_cfg["transcript_dir"] = expand_path(transcript_dir)
    else:
        fd_cfg["transcript_dir"] = outputs_cfg["format_dirs"].get(transcript_format)

    metadata_csv = fd_cfg.get("metadata_csv")
    if metadata_csv:
        fd_cfg["metadata_csv"] = expand_path(metadata_csv)

    return cfg


def ensure_directories(cfg: Dict, rank: int) -> None:
    if rank != 0:
        return
    os.makedirs(cfg["frame_descriptions"]["output_dir"], exist_ok=True)


def discover_frame_files(frames_root: str, image_extension: str) -> Dict[str, List[Tuple[str, str]]]:
    extension = image_extension.lstrip(".")
    pattern = os.path.join(frames_root, "**", f"*.{extension}")
    files = glob.glob(pattern, recursive=True)
    files.sort()

    frame_map: Dict[str, List[Tuple[str, str]]] = {}
    for path in files:
        rel = os.path.relpath(path, frames_root)
        stem = os.path.splitext(os.path.basename(path))[0]
        if "_t" not in stem:
            continue
        item_id = stem.rsplit("_t", 1)[0]
        frame_map.setdefault(item_id, []).append((path, rel))
    return frame_map


def load_metadata(cfg: Dict) -> Dict[str, Dict[str, str]]:
    metadata_path = cfg.get("metadata_csv")
    if not metadata_path:
        return {}

    id_column = cfg.get("metadata_id_column", "ID")
    fields: Sequence[str] = cfg.get("metadata_fields", [])

    metadata: Dict[str, Dict[str, str]] = {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if id_column not in reader.fieldnames:
            raise ValueError(
                f"Metadata CSV missing id column '{id_column}'. "
                f"Columns available: {reader.fieldnames}",
            )
        missing = [field for field in fields if field not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Metadata CSV missing requested fields: {missing}. "
                f"Columns available: {reader.fieldnames}",
            )
        for row in reader:
            identifier = row[id_column]
            metadata[identifier] = {field: row[field] for field in fields}
    return metadata


def load_transcript_text(item_id: str, cfg: Dict) -> str:
    transcript_dir = cfg.get("transcript_dir")
    transcript_format = cfg.get("transcript_format", "txt")
    fallback = cfg.get("transcript_missing_fallback", "Transcript unavailable.")
    if not transcript_dir:
        return fallback

    path = os.path.join(transcript_dir, f"{item_id}.{transcript_format}")
    if not os.path.exists(path):
        return fallback

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    max_chars = cfg.get("max_transcript_chars")
    if max_chars and len(text) > max_chars:
        trunc_notice = "\n\n[Transcript truncated for frame description generation.]"
        text = text[: max_chars - len(trunc_notice)] + trunc_notice

    return text.strip()


def construct_metadata_block(metadata: Dict[str, Dict[str, str]], item_id: str, fallback: str) -> str:
    if item_id not in metadata:
        return fallback
    entries = []
    for key, value in metadata[item_id].items():
        if not value:
            continue
        pretty_key = key.replace("_", " ").title()
        entries.append(f"{pretty_key}: {value}")
    return "\n".join(entries) if entries else fallback


def parse_timestamp_from_filename(rel_path: str) -> Tuple[str, float]:
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    ts_part = stem.rsplit("_t", 1)[-1]
    try:
        milliseconds = int(ts_part)
    except ValueError:
        milliseconds = 0
    seconds = milliseconds / 1000.0
    return stem, seconds


def encode_image_as_base64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def build_prompt(
    frame_cfg: Dict,
    metadata_block: str,
    transcript: str,
    timestamp_seconds: float,
) -> Dict[str, str]:
    prompt_template = frame_cfg.get(
        "prompt_template",
        (
            "Describe what is depicted in this archival video frame in no more than 20 words.\n"
            "{metadata_block}\nTranscript:\n{transcript}\nTimestamp: {timestamp_seconds:.2f} seconds."
        ),
    )
    user_prompt = prompt_template.format(
        metadata_block=metadata_block,
        transcript=transcript,
        timestamp_seconds=timestamp_seconds,
    )
    system_prompt = frame_cfg.get(
        "system_prompt",
        (
            "You are an archivist describing individual frames from historical audio-visual materials. "
            "Focus on neutral, factual, concise descriptions and include any visible text."
        ),
    )
    return {"system": system_prompt, "user": user_prompt}


def create_openai_client(frame_cfg: Dict) -> OpenAI:
    api_key_env = frame_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            "Set it to your OpenAI API key before running frame descriptions.",
        )
    return OpenAI(api_key=api_key)


def describe_frame(
    client: OpenAI,
    image_b64: str,
    frame_cfg: Dict,
    metadata_block: str,
    transcript: str,
    timestamp_seconds: float,
) -> str:
    prompts = build_prompt(frame_cfg, metadata_block, transcript, timestamp_seconds)
    response = client.responses.create(
        model=frame_cfg.get("model", "gpt-4o"),
        input=[
            {"role": "system", "content": prompts["system"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts["user"]},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
        max_output_tokens=120,
        temperature=frame_cfg.get("temperature", 0.2),
    )
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel GPT-Vision descriptions of keyframes extracted from audiovisual collections.",
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
    frame_cfg = config["frame_descriptions"]
    ensure_directories(config, rank)

    if rank == 0:
        frame_map = discover_frame_files(frame_cfg["frames_root"], frame_cfg.get("image_extension", "jpg"))
        if not frame_map:
            sys.exit(f"No keyframes found under {frame_cfg['frames_root']}.")
        metadata = load_metadata(frame_cfg)
    else:
        frame_map = None
        metadata = None

    frame_map = comm.bcast(frame_map, root=0)
    metadata = comm.bcast(metadata, root=0)

    item_ids = sorted(frame_map.keys())
    items_for_rank = np.array_split(item_ids, size)[rank]
    items_for_rank = list(items_for_rank)

    if rank == 0:
        print(
            f"Starting frame description generation with {size} MPI ranks, "
            f"{len(item_ids)} videos worth of keyframes, model '{frame_cfg.get('model', 'gpt-4o')}'.",
            flush=True,
        )

    client = create_openai_client(frame_cfg)
    metadata_fallback = frame_cfg.get("metadata_missing_fallback", "Archival metadata unavailable.")
    output_dir = frame_cfg["output_dir"]
    overwrite = frame_cfg.get("overwrite", False)
    max_frames_per_item = frame_cfg.get("max_frames_per_item")

    transcript_cache: Dict[str, str] = {}
    processed_frames = 0
    started_at = time.time()
    errors: List[Dict[str, str]] = []

    for item_id in items_for_rank:
        frames = frame_map[item_id]
        if max_frames_per_item is not None:
            frames = frames[:max_frames_per_item]

        transcript = transcript_cache.get(item_id)
        if transcript is None:
            transcript = load_transcript_text(item_id, frame_cfg)
            transcript_cache[item_id] = transcript

        metadata_block = construct_metadata_block(metadata, item_id, metadata_fallback)

        for frame_path, rel_path in frames:
            stem, timestamp_seconds = parse_timestamp_from_filename(rel_path)
            rel_no_ext = os.path.splitext(rel_path)[0]
            out_path = os.path.join(output_dir, rel_no_ext + ".txt")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if os.path.exists(out_path) and not overwrite:
                continue

            try:
                image_b64 = encode_image_as_base64(frame_path)
                description = describe_frame(
                    client,
                    image_b64,
                    frame_cfg,
                    metadata_block,
                    transcript,
                    timestamp_seconds,
                )
                with open(out_path, "w", encoding="utf-8") as handle:
                    handle.write(description)
                processed_frames += 1
            except Exception as exc:  # noqa: BLE001
                errors.append({"path": frame_path, "error": str(exc)})
                if config.get("logging", {}).get("verbose"):
                    print(
                        f"[rank {rank}] Error describing {frame_path}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        interval = config.get("logging", {}).get("progress_interval", 5)
        if interval and processed_frames and processed_frames % interval == 0:
            elapsed = time.time() - started_at
            rate = processed_frames / elapsed if elapsed else 0.0
            print(
                f"[rank {rank}] Generated {processed_frames} frame descriptions "
                f"({rate:.2f} items/sec).",
                flush=True,
            )

    gathered_errors = comm.gather(errors, root=0)
    gathered_counts = comm.gather(processed_frames, root=0)

    if rank == 0:
        total_processed = sum(gathered_counts)
        print(
            f"Frame description generation complete. Wrote {total_processed} descriptions "
            f"across {size} ranks at {datetime.now().isoformat()}",
            flush=True,
        )
        aggregated_errors = [item for sublist in gathered_errors for item in sublist]
        if aggregated_errors:
            print(
                f"{len(aggregated_errors)} frames encountered errors. See details below:",
                file=sys.stderr,
            )
            for err in aggregated_errors:
                print(
                    f" - {err['path']}: {err['error']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()

