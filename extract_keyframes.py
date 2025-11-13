import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml
from mpi4py import MPI


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping.")
    return config


def validate_config(cfg: Dict) -> Dict:
    keyframe_cfg = cfg.setdefault("keyframes", {})
    if not keyframe_cfg.get("enabled", False):
        raise ValueError(
            "Keyframe extraction is disabled in the configuration. "
            "Set `keyframes.enabled` to true.",
        )

    input_cfg = cfg.setdefault("input", {})
    media_root = input_cfg.get("media_root")
    if not media_root:
        raise ValueError("`input.media_root` is required.")
    input_cfg["media_root"] = expand_path(media_root)

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

    keyframe_cfg.setdefault("modes", ["speech_segments", "interval"])
    keyframe_cfg.setdefault("output_dir", "keyframes")
    keyframe_cfg["output_dir"] = expand_path(keyframe_cfg["output_dir"])

    if keyframe_cfg.get("speech_segment_dir"):
        keyframe_cfg["speech_segment_dir"] = expand_path(keyframe_cfg["speech_segment_dir"])
    else:
        seg_fmt = keyframe_cfg.get("speech_segment_format", "tsv")
        keyframe_cfg["speech_segment_dir"] = outputs_cfg["format_dirs"].get(seg_fmt)

    return cfg


def ensure_directories(cfg: Dict, rank: int) -> None:
    if rank != 0:
        return
    key_cfg = cfg["keyframes"]
    os.makedirs(key_cfg["output_dir"], exist_ok=True)
    for mode in key_cfg.get("modes", []):
        os.makedirs(os.path.join(key_cfg["output_dir"], mode), exist_ok=True)


def discover_media_files(cfg: Dict) -> List[str]:
    input_cfg = cfg["input"]
    media_root = input_cfg["media_root"]
    exts = {ext.lower() for ext in input_cfg.get("include_extensions", [])}
    if not exts:
        exts = {".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".mp3", ".wav", ".m4a"}

    recurse = input_cfg.get("recurse", False)
    filepaths: List[str] = []
    if recurse:
        for root, _, files in os.walk(media_root):
            for fname in files:
                if os.path.splitext(fname)[1].lower() in exts:
                    filepaths.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(media_root):
            fpath = os.path.join(media_root, fname)
            if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in exts:
                filepaths.append(fpath)

    filepaths.sort()
    return filepaths


def safe_output_stem(media_path: str, media_root: str) -> str:
    rel = os.path.relpath(media_path, media_root)
    stem, _ = os.path.splitext(rel)
    return stem.replace(os.sep, "__")


def ffprobe_duration_seconds(media_path: str) -> float:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to determine video duration but was not found on PATH.")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        media_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media_path}: {result.stderr}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"Unable to parse ffprobe duration for {media_path}: {result.stdout}") from exc


def generate_interval_timestamps(duration: float, cfg: Dict) -> List[float]:
    interval = float(cfg.get("interval_seconds", 3.0))
    if interval <= 0:
        return []
    max_frames = cfg.get("max_interval_frames")
    times = []
    current = interval
    while current < duration:
        times.append(current)
        if max_frames is not None and len(times) >= max_frames:
            break
        current += interval
    return times


def load_speech_segments_midpoints(stem: str, cfg: Dict) -> List[float]:
    segment_dir = cfg["speech_segment_dir"]
    if not segment_dir:
        return []
    seg_format = cfg.get("speech_segment_format", "tsv")
    path = os.path.join(segment_dir, f"{stem}.{seg_format}")
    if not os.path.exists(path):
        return []

    midpoints: List[float] = []
    max_frames = cfg.get("max_speech_frames")

    if seg_format == "tsv":
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                try:
                    start = float(row.get("start", "0"))
                    end = float(row.get("end", "0"))
                except ValueError:
                    continue
                midpoint = start + (end - start) / 2.0
                midpoints.append(midpoint)
                if max_frames is not None and len(midpoints) >= max_frames:
                    break
    elif seg_format == "json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        segments = data.get("segments", [])
        for seg in segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            midpoint = start + (end - start) / 2.0
            midpoints.append(midpoint)
            if max_frames is not None and len(midpoints) >= max_frames:
                break
    else:
        raise ValueError(f"Unsupported speech segment format: {seg_format}")

    return midpoints


def deduplicate_sorted(times: Iterable[float]) -> List[float]:
    unique = sorted(set(times))
    return [t for t in unique if t >= 0]


def timestamp_to_filename_component(timestamp_seconds: float) -> str:
    milliseconds = int(round(timestamp_seconds * 1000))
    return f"t{milliseconds:09d}"


def extract_frame(
    media_path: str,
    timestamp: float,
    output_path: str,
    cfg: Dict,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for frame extraction but was not found on PATH.")

    image_ext = cfg.get("image_extension", "jpg")
    jpeg_quality = cfg.get("jpeg_quality", 2)
    additional_args = cfg.get("ffmpeg_additional_args", [])

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        media_path,
        "-frames:v",
        "1",
    ]
    if image_ext.lower() in {"jpg", "jpeg"}:
        cmd.extend(["-qscale:v", str(jpeg_quality)])
    cmd.extend(additional_args)
    cmd.append(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {media_path} @ {timestamp:.3f}s: {result.stderr.decode()}")


def process_media(
    media_path: str,
    cfg: Dict,
    stem: str,
) -> int:
    key_cfg = cfg["keyframes"]
    media_root = cfg["input"]["media_root"]
    overwrite = key_cfg.get("overwrite", False)
    modes = key_cfg.get("modes", [])

    timestamps: List[float] = []
    mode_timestamp_map: Dict[str, List[float]] = {}
    if "interval" in modes:
        duration = ffprobe_duration_seconds(media_path)
        interval_times = generate_interval_timestamps(duration, key_cfg)
        timestamps.extend(interval_times)
        mode_timestamp_map["interval"] = interval_times
    if "speech_segments" in modes:
        speech_times = load_speech_segments_midpoints(stem, key_cfg)
        timestamps.extend(speech_times)
        mode_timestamp_map["speech_segments"] = speech_times

    timestamps = deduplicate_sorted(timestamps)
    if not timestamps:
        return 0

    max_total = key_cfg.get("max_total_frames")
    if max_total is not None:
        timestamps = timestamps[:max_total]
    timestamps_set = set(timestamps)

    extracted = 0
    for mode, mode_times in mode_timestamp_map.items():
        if not mode_times:
            continue
        mode_dir = os.path.join(key_cfg["output_dir"], mode)
        for ts in mode_times:
            if ts not in timestamps_set:
                continue
            ts_tag = timestamp_to_filename_component(ts)
            output_filename = f"{stem}_{ts_tag}.{key_cfg.get('image_extension', 'jpg')}"
            output_path = os.path.join(mode_dir, output_filename)

            if os.path.exists(output_path) and not overwrite:
                continue

            extract_frame(media_path, ts, output_path, key_cfg)
            extracted += 1

    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel keyframe extraction for large audiovisual collections.",
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
    ensure_directories(config, rank)

    if rank == 0:
        media_files = discover_media_files(config)
        if not media_files:
            sys.exit(f"No media files found in {config['input']['media_root']}.")
    else:
        media_files = None

    media_files = comm.bcast(media_files, root=0)
    files_for_rank = np.array_split(media_files, size)[rank]
    files_for_rank = list(files_for_rank)

    if rank == 0:
        print(
            f"Starting keyframe extraction with {size} MPI ranks, "
            f"{len(media_files)} media files.",
            flush=True,
        )

    errors: List[Dict[str, str]] = []
    extracted_total = 0
    started_at = time.time()

    for idx, media_path in enumerate(files_for_rank, start=1):
        stem = safe_output_stem(media_path, config["input"]["media_root"])
        try:
            extracted = process_media(media_path, config, stem)
            extracted_total += extracted
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": media_path, "error": str(exc)})
            if config.get("logging", {}).get("verbose"):
                print(
                    f"[rank {rank}] Error extracting frames from {media_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        interval = config.get("logging", {}).get("progress_interval", 5)
        if interval and idx % interval == 0:
            elapsed = time.time() - started_at
            rate = extracted_total / elapsed if elapsed else 0.0
            print(
                f"[rank {rank}] Processed {idx}/{len(files_for_rank)} videos; "
                f"frames extracted so far: {extracted_total} "
                f"({rate:.2f} frames/sec).",
                flush=True,
            )

    gathered_errors = comm.gather(errors, root=0)
    gathered_counts = comm.gather(extracted_total, root=0)

    if rank == 0:
        total_frames = sum(gathered_counts)
        print(
            f"Keyframe extraction complete. Generated {total_frames} frames "
            f"across {size} ranks at {datetime.now().isoformat()}",
            flush=True,
        )
        aggregated_errors = [item for sublist in gathered_errors for item in sublist]
        if aggregated_errors:
            print(
                f"{len(aggregated_errors)} videos encountered errors. See details below:",
                file=sys.stderr,
            )
            for err in aggregated_errors:
                print(
                    f" - {err['path']}: {err['error']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()

