import argparse
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from typing import Dict, List, Sequence

import numpy as np
import whisper
from whisper.utils import get_writer
import yaml
from mpi4py import MPI
import subprocess


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a mapping.")
    return config


def validate_and_prepare_paths(config: Dict, rank: int) -> Dict:
    cfg = config.copy()

    input_cfg = cfg.setdefault("input", {})
    media_root = input_cfg.get("media_root")
    if not media_root:
        raise ValueError("`input.media_root` is required.")
    media_root = expand_path(media_root)
    if rank == 0 and not os.path.exists(media_root):
        raise FileNotFoundError(f"Media root does not exist: {media_root}")
    input_cfg["media_root"] = media_root

    output_cfg = cfg.setdefault("outputs", {})
    base_dir = output_cfg.get("base_dir", "transcripts")
    base_dir = expand_path(base_dir)
    output_cfg["base_dir"] = base_dir

    formats: Sequence[str] = output_cfg.get("formats", ["txt"])
    if not formats:
        raise ValueError("`outputs.formats` must contain at least one entry.")
    format_dirs = {}
    for fmt in formats:
        format_dirs[fmt] = os.path.join(base_dir, fmt)
    output_cfg["format_dirs"] = format_dirs

    preprocess_cfg = cfg.setdefault("preprocessing", {})
    if preprocess_cfg.get("extract_audio", False):
        audio_dir = preprocess_cfg.get("audio_dir", "preprocessed_audio")
        audio_dir = expand_path(audio_dir)
        preprocess_cfg["audio_dir"] = audio_dir
    else:
        preprocess_cfg["audio_dir"] = None

    return cfg


def ensure_directories(cfg: Dict, rank: int) -> None:
    if rank != 0:
        return

    output_cfg = cfg["outputs"]
    os.makedirs(output_cfg["base_dir"], exist_ok=True)
    for fmt_dir in output_cfg["format_dirs"].values():
        os.makedirs(fmt_dir, exist_ok=True)

    preprocess_cfg = cfg["preprocessing"]
    if preprocess_cfg.get("extract_audio") and preprocess_cfg.get("audio_dir"):
        os.makedirs(preprocess_cfg["audio_dir"], exist_ok=True)


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
    stem = stem.replace(os.sep, "__")
    return stem


def audio_output_path(preprocess_cfg: Dict, stem: str) -> str:
    audio_dir = preprocess_cfg["audio_dir"]
    desired_ext = preprocess_cfg.get("audio_extension")
    if not desired_ext:
        codec = preprocess_cfg.get("audio_codec", "pcm_s16le")
        desired_ext = ".wav" if codec.startswith("pcm") else ".mp3"
    if not desired_ext.startswith("."):
        desired_ext = f".{desired_ext}"
    return os.path.join(audio_dir, stem + desired_ext)


def normalize_audio(
    media_path: str,
    preprocess_cfg: Dict,
    stem: str,
    overwrite: bool,
) -> str:
    output_path = audio_output_path(preprocess_cfg, stem)
    if os.path.exists(output_path) and not overwrite:
        return output_path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for audio extraction but was not found on PATH.")

    sample_rate = str(preprocess_cfg.get("sample_rate", 16000))
    codec = preprocess_cfg.get("audio_codec", "pcm_s16le")
    channels = str(preprocess_cfg.get("channels", 1))
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        media_path,
        "-vn",
        "-acodec",
        codec,
        "-ar",
        sample_rate,
        "-ac",
        channels,
        output_path,
    ]
    proc = subprocess.run(
        ffmpeg_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {media_path}:\n{proc.stderr}")

    return output_path


def choose_device(model_name: str, device_cfg: str) -> whisper.Whisper:
    if device_cfg and device_cfg.lower() not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be one of 'auto', 'cpu', or 'cuda'.")
    kwargs = {}
    if device_cfg and device_cfg.lower() in {"cpu", "cuda"}:
        kwargs["device"] = device_cfg.lower()
    return whisper.load_model(model_name, **kwargs)


def build_transcribe_kwargs(transcription_cfg: Dict) -> Dict:
    keys = [
        "language",
        "temperature",
        "condition_on_previous_text",
        "initial_prompt",
        "suppress_tokens",
    ]
    kwargs = {}
    for key in keys:
        if key in transcription_cfg and transcription_cfg[key] is not None:
            kwargs[key] = transcription_cfg[key]
    if "fp16" in transcription_cfg:
        kwargs["fp16"] = transcription_cfg["fp16"]
    else:
        device = transcription_cfg.get("device", "auto")
        kwargs["fp16"] = device == "cuda"
    return kwargs


def writer_extension(fmt: str) -> str:
    mapping = {
        "txt": ".txt",
        "json": ".json",
        "tsv": ".tsv",
        "srt": ".srt",
        "vtt": ".vtt",
    }
    if fmt not in mapping:
        raise ValueError(f"Unsupported output format: {fmt}")
    return mapping[fmt]


def process_media_file(
    model: whisper.Whisper,
    media_path: str,
    cfg: Dict,
    stem: str,
    writer_options: Dict,
) -> Dict:
    preprocess_cfg = cfg["preprocessing"]
    outputs_cfg = cfg["outputs"]
    transcription_cfg = cfg["transcription"]

    media_to_transcribe = media_path
    if preprocess_cfg.get("extract_audio"):
        media_to_transcribe = normalize_audio(
            media_path,
            preprocess_cfg,
            stem,
            overwrite=preprocess_cfg.get("overwrite_audio", False),
        )

    kwargs = build_transcribe_kwargs(transcription_cfg)
    result = model.transcribe(media_to_transcribe, **kwargs)

    for fmt in outputs_cfg["formats"]:
        fmt_dir = outputs_cfg["format_dirs"][fmt]
        out_fname = stem + writer_extension(fmt)
        writer = get_writer(fmt, fmt_dir)
        writer(result, out_fname, writer_options)

    return {
        "duration": result.get("duration"),
        "language": result.get("language"),
        "segments": len(result.get("segments", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel transcription for large audiovisual collections using Whisper.",
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
            config = validate_and_prepare_paths(config, rank=0)
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

    transcription_cfg = config.setdefault("transcription", {})
    device_cfg = transcription_cfg.get("device", "auto")
    model_name = transcription_cfg.get("model", "large-v3")
    if rank == 0:
        print(
            f"Starting transcription with {size} MPI ranks, "
            f"{len(media_files)} media files, Whisper model '{model_name}'.",
            flush=True,
        )

    model = choose_device(model_name, device_cfg)
    writer_options = config["outputs"].get("writer_options", {})
    sentinel_fmt = config["outputs"]["formats"][0]
    sentinel_dir = config["outputs"]["format_dirs"][sentinel_fmt]
    sentinel_ext = writer_extension(sentinel_fmt)
    overwrite_transcripts = config["outputs"].get("overwrite", False)

    media_root = config["input"]["media_root"]
    errors: List[Dict[str, str]] = []
    processed = 0
    started_at = time.time()

    for idx, media_path in enumerate(files_for_rank, start=1):
        stem = safe_output_stem(media_path, media_root)
        sentinel_path = os.path.join(sentinel_dir, stem + sentinel_ext)
        if not overwrite_transcripts and os.path.exists(sentinel_path):
            continue

        try:
            process_media_file(model, media_path, config, stem, writer_options)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": media_path, "error": str(exc)})
            if config.get("logging", {}).get("verbose"):
                print(
                    f"[rank {rank}] Error processing {media_path}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        interval = config.get("logging", {}).get("progress_interval", 5)
        if interval and processed and processed % interval == 0:
            elapsed = time.time() - started_at
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"[rank {rank}] Completed {processed} files "
                f"({idx}/{len(files_for_rank)}) at {rate:.2f} files/sec.",
                flush=True,
            )

    gathered_errors = comm.gather(errors, root=0)
    gathered_counts = comm.gather(processed, root=0)

    if rank == 0:
        total_processed = sum(gathered_counts)
        print(
            f"Transcription complete. Processed {total_processed} files "
            f"across {size} ranks at {datetime.now().isoformat()}",
            flush=True,
        )
        aggregated_errors = [item for sublist in gathered_errors for item in sublist]
        if aggregated_errors:
            print(
                f"{len(aggregated_errors)} files encountered errors. See details below:",
                file=sys.stderr,
            )
            for err in aggregated_errors:
                print(
                    f" - {err['path']}: {err['error']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()

