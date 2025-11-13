import argparse
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
    ac_cfg = cfg.setdefault("accessibility", {})
    if not ac_cfg.get("enabled", False):
        raise ValueError(
            "Accessibility generation is disabled in the configuration. "
            "Set `accessibility.enabled` to true.",
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

    transcript_format = ac_cfg.get("transcript_format", "txt")
    transcripts_subdir = ac_cfg.get("transcripts_subdir")
    if transcripts_subdir:
        ac_cfg["transcript_dir"] = expand_path(transcripts_subdir)
    else:
        ac_cfg["transcript_dir"] = outputs_cfg["format_dirs"].get(transcript_format)

    if ac_cfg["transcript_dir"] is None:
        raise ValueError(
            "Unable to locate transcripts for accessibility generation. "
            "Specify `accessibility.transcripts_subdir` or ensure the format is in outputs.formats.",
        )

    if ac_cfg.get("frame_descriptions_dirs"):
        ac_cfg["frame_descriptions_dirs"] = [
            expand_path(path) for path in ac_cfg["frame_descriptions_dirs"]
        ]
    else:
        frame_cfg = cfg.get("frame_descriptions", {})
        if frame_cfg.get("enabled"):
            ac_cfg["frame_descriptions_dirs"] = [
                expand_path(frame_cfg.get("output_dir", "frame_descriptions")),
            ]
        else:
            ac_cfg["frame_descriptions_dirs"] = []

    ac_cfg["output_dir"] = expand_path(ac_cfg.get("output_dir", "accessibility_notes"))

    return cfg


def ensure_directories(cfg: Dict, rank: int) -> None:
    if rank != 0:
        return
    os.makedirs(cfg["accessibility"]["output_dir"], exist_ok=True)


def discover_transcripts(directory: str, extension: str) -> List[str]:
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.endswith(f".{extension}")
    ]
    files.sort()
    return files


def read_transcript(path: str, max_chars: Optional[int]) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if max_chars and len(text) > max_chars:
        notice = "\n\n[Transcript truncated for accessibility generation.]"
        text = text[: max_chars - len(notice)] + notice
    return text.strip()


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


def gather_frame_descriptions(item_id: str, cfg: Dict) -> List[Tuple[float, str]]:
    directories = cfg.get("frame_descriptions_dirs", [])
    if not directories:
        return []
    extension = cfg.get("frame_description_extension", "txt").lstrip(".")
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
    return entries


def format_frame_descriptions(frames: List[Tuple[float, str]], cfg: Dict) -> str:
    if not frames:
        return cfg.get(
            "frame_descriptions_missing_fallback",
            "No visual descriptions available.",
        )

    entry_template = cfg.get("frame_entry_template", "{timestamp:.1f}s: {description}")
    lines = [entry_template.format(timestamp=t, description=d) for t, d in frames]
    return "\n".join(lines)


def create_openai_client(cfg: Dict) -> OpenAI:
    api_key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable '{api_key_env}' is not set. "
            "Set it to your OpenAI API key before running accessibility generation.",
        )
    return OpenAI(api_key=api_key)


def generate_accessibility_note(
    client: OpenAI,
    transcript_text: str,
    frame_block: str,
    cfg: Dict,
) -> str:
    prompt_template = cfg.get(
        "prompt_template",
        (
            "Create concise audio-description narration cues for the following item.\n"
            "Transcript:\n{transcript}\nVisual descriptions:\n{frame_descriptions}"
        ),
    )
    user_prompt = prompt_template.format(
        transcript=transcript_text,
        frame_descriptions=frame_block,
    )
    system_prompt = cfg.get(
        "system_prompt",
        (
            "You produce accessible narration cues that describe observable visual content, "
            "speaker changes, and on-screen text in a neutral tone."
        ),
    )

    response = client.responses.create(
        model=cfg.get("model", "gpt-4o-mini"),
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_output_tokens=800,
    )
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audio-description style accessibility notes from transcripts and frame descriptions.",
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
    ac_cfg = config["accessibility"]
    ensure_directories(config, rank)

    if rank == 0:
        transcript_dir = ac_cfg["transcript_dir"]
        transcript_format = ac_cfg.get("transcript_format", "txt")
        transcript_files = discover_transcripts(transcript_dir, transcript_format)
        if not transcript_files:
            sys.exit(f"No transcripts found in {transcript_dir} with extension '.{transcript_format}'.")
        max_chars = ac_cfg.get("max_transcript_chars")
    else:
        transcript_files = None
        max_chars = None

    transcript_files = comm.bcast(transcript_files, root=0)
    max_chars = comm.bcast(max_chars, root=0)

    files_for_rank = np.array_split(transcript_files, size)[rank]
    files_for_rank = list(files_for_rank)

    if rank == 0:
        print(
            f"Starting accessibility generation with {size} MPI ranks, "
            f"{len(transcript_files)} transcripts, "
            f"model '{ac_cfg.get('model', 'gpt-4o-mini')}'.",
            flush=True,
        )

    client = create_openai_client(ac_cfg)
    output_dir = ac_cfg["output_dir"]
    overwrite = ac_cfg.get("overwrite", False)

    processed = 0
    started_at = time.time()
    errors: List[Dict[str, str]] = []

    for idx, transcript_path in enumerate(files_for_rank, start=1):
        item_id = os.path.splitext(os.path.basename(transcript_path))[0]
        output_path = os.path.join(output_dir, f"{item_id}.txt")

        if not overwrite and os.path.exists(output_path):
            continue

        try:
            transcript_text = read_transcript(transcript_path, max_chars)
            frames = gather_frame_descriptions(item_id, ac_cfg)
            frame_block = format_frame_descriptions(frames, ac_cfg)
            note = generate_accessibility_note(client, transcript_text, frame_block, ac_cfg)
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(note)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": transcript_path, "error": str(exc)})
            if ac_cfg.get("verbose"):
                print(
                    f"[rank {rank}] Error generating accessibility note for {transcript_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        interval = config.get("logging", {}).get("progress_interval", 5)
        if interval and processed and processed % interval == 0:
            elapsed = time.time() - started_at
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"[rank {rank}] Generated {processed} accessibility notes "
                f"({idx}/{len(files_for_rank)}) at {rate:.2f} items/sec.",
                flush=True,
            )

    gathered_errors = comm.gather(errors, root=0)
    gathered_counts = comm.gather(processed, root=0)

    if rank == 0:
        total_processed = sum(gathered_counts)
        print(
            f"Accessibility generation complete. Created {total_processed} notes "
            f"across {size} ranks at {datetime.now().isoformat()}",
            flush=True,
        )
        aggregated_errors = [item for sublist in gathered_errors for item in sublist]
        if aggregated_errors:
            print(
                f"{len(aggregated_errors)} items encountered errors. See details below:",
                file=sys.stderr,
            )
            for err in aggregated_errors:
                print(
                    f" - {err['path']}: {err['error']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()

