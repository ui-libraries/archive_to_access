import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def prompt_bool(question: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    answer = input(question + suffix + " ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def prompt_int(question: str, default: int) -> int:
    answer = input(f"{question} [{default}] ").strip()
    if not answer:
        return default
    try:
        return int(answer)
    except ValueError:
        print("Invalid integer; using default.")
        return default


def prompt_str(question: str, default: str) -> str:
    answer = input(f"{question} [{default}] ").strip()
    return answer or default


def main() -> None:
    print("Parallel AV Toolkit configuration wizard\n")
    config_path = input("Where should the config be written? [config.yaml] ").strip() or "config.yaml"

    media_root = input("Path to your media collection: ").strip()
    if not media_root:
        print("Media root is required.")
        return

    config: Dict[str, Any] = {}
    config["input"] = {
        "media_root": media_root,
        "include_extensions": [".mp4", ".mov", ".mkv", ".mp3", ".wav", ".m4a"],
        "recurse": prompt_bool("Search subdirectories for media files?", True),
    }

    config["preprocessing"] = {
        "extract_audio": prompt_bool("Extract/normalize audio with ffmpeg before transcription?", True),
        "audio_dir": "preprocessed_audio",
        "overwrite_audio": False,
        "sample_rate": 16000,
        "audio_codec": "pcm_s16le",
        "channels": 1,
    }

    whisper_model = prompt_str("Whisper model (tiny/base/small/medium/large-v3)", "large-v3")
    config["transcription"] = {
        "model": whisper_model,
        "device": prompt_str("Preferred device (auto/cuda/cpu)", "auto"),
        "language": None,
        "temperature": 0.0,
        "condition_on_previous_text": True,
        "initial_prompt": None,
        "suppress_tokens": "-1",
    }

    config["outputs"] = {
        "base_dir": "transcripts",
        "formats": ["txt", "json", "tsv"],
        "overwrite": False,
        "writer_options": {
            "max_line_width": None,
            "max_line_count": None,
            "highlight_words": False,
        },
    }

    enable_keyframes = prompt_bool("Generate still keyframes?", True)
    config["keyframes"] = {
        "enabled": enable_keyframes,
        "modes": ["speech_segments", "interval"],
        "output_dir": "keyframes",
        "overwrite": False,
        "image_extension": "jpg",
        "jpeg_quality": 2,
        "interval_seconds": 3.0,
        "speech_segment_format": "tsv",
    }

    enable_frame_desc = prompt_bool("Describe frames with GPT-Vision?", enable_keyframes)
    config["frame_descriptions"] = {
        "enabled": enable_frame_desc,
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "frames_root": None,
        "output_dir": "frame_descriptions",
        "overwrite": False,
        "max_frames_per_item": 40,
        "max_transcript_chars": 6000,
        "temperature": 0.2,
        "metadata_csv": None,
        "metadata_id_column": "ID",
        "metadata_fields": ["ELECTION", "PARTY", "FIRST_NAME", "LAST_NAME", "TITLE"],
        "transcript_format": "txt",
        "transcripts_subdir": None,
    }

    enable_summaries = prompt_bool("Generate GPT summaries?", True)
    config["summarization"] = {
        "enabled": enable_summaries,
        "model": "gpt-4o-mini",
        "word_limit": prompt_int("Summary word limit", 60),
        "system_prompt": "You are an assistant that writes concise, neutral descriptions of archival audiovisual materials.",
        "prompt_template": "Provide a {word_limit}-word summary.\n{metadata_block}\nTranscript:\n{transcript}\nVisual descriptions:\n{frame_descriptions}",
        "transcript_format": "txt",
        "transcripts_subdir": None,
        "max_transcript_chars": 12000,
        "output_dir": "summaries",
        "overwrite": False,
        "api_key_env": "OPENAI_API_KEY",
        "metadata_csv": None,
        "metadata_id_column": "ID",
        "metadata_fields": ["ELECTION", "PARTY", "FIRST_NAME", "LAST_NAME", "TITLE"],
        "metadata_missing_fallback": "Archival metadata unavailable.",
        "use_frame_descriptions": enable_frame_desc,
        "frame_descriptions_dirs": None,
        "frame_descriptions_limit": 20,
        "frame_entry_template": "{idx}. ({timestamp:.1f}s) {description}",
        "frame_block_template": "Visual key points:\n{frames}",
        "frame_descriptions_missing_fallback": "No frame descriptions available.",
        "temperature": 0.2,
    }

    # Optional sections inherited from sample config
    config["tagging"] = {
        "enabled": prompt_bool("Generate entity/topic tags?", True),
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "transcript_format": "txt",
        "transcripts_subdir": None,
        "output_dir": "tags",
        "overwrite": False,
        "max_transcript_chars": 8000,
        "metadata_csv": None,
        "metadata_id_column": "ID",
        "metadata_fields": ["ELECTION", "PARTY", "FIRST_NAME", "LAST_NAME", "TITLE"],
    }

    config["collection_reports"] = {
        "enabled": prompt_bool("Create collection-level reports?", True),
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "summaries_dir": "summaries",
        "tags_dir": "tags",
        "output_path": "collection_report.md",
        "max_items": 200,
    }

    config["accessibility"] = {
        "enabled": prompt_bool("Generate accessibility narration cues?", True),
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "transcript_format": "txt",
        "frame_descriptions_dirs": None,
        "output_dir": "accessibility_notes",
        "overwrite": False,
        "max_transcript_chars": 8000,
    }

    config["preview_dashboard"] = {
        "enabled": True,
        "output_path": "preview/index.html",
        "max_items": 50,
        "include_frames": True,
        "include_summaries": True,
        "include_tags": True,
        "include_transcript": False,
        "static_assets_dir": "preview/assets",
        "css_file": None,
    }

    config["quality_control"] = {
        "enabled": True,
        "output_csv": "quality_metrics.csv",
        "summary_dir": "summaries",
        "transcript_dir": None,
        "transcript_format": "txt",
        "frame_descriptions_dirs": None,
        "min_summary_words": 30,
        "max_summary_words": 70,
    }

    config["provenance"] = {
        "enabled": True,
        "log_path": "provenance_log.jsonl",
    }

    config["iiif"] = {
        "enabled": prompt_bool("Produce an IIIF manifest?", False),
        "manifest_output": "iiif/manifest.json",
        "frames_root": None,
        "base_media_url": prompt_str("Base media URL for IIIF (optional)", "https://example.org/media"),
        "base_frame_url": prompt_str("Base frame URL for IIIF (optional)", "https://example.org/frames"),
        "metadata_fields": ["ELECTION", "PARTY", "TITLE"],
    }

    config["catalog_export"] = {
        "enabled": True,
        "output_csv": "catalog_export.csv",
        "include_transcript_excerpt": True,
        "transcript_excerpt_chars": 500,
        "include_summary": True,
        "include_tags": True,
    }

    config["search_index"] = {
        "enabled": True,
        "sqlite_path": "search_index.db",
        "include_transcripts": True,
        "include_summaries": True,
        "include_tags": True,
    }

    config["clustering"] = {
        "enabled": prompt_bool("Cluster frame descriptions to spot visual themes?", False),
        "model": "text-embedding-3-small",
        "api_key_env": "OPENAI_API_KEY",
        "frame_descriptions_dirs": None,
        "output_dir": "clusters",
        "overwrite": False,
        "n_clusters": 20,
    }

    config["workflow"] = {
        "steps": [
            "transcribe",
            "keyframes",
            "describe_frames",
            "summarize",
            "tags",
            "accessibility",
            "collection_report",
            "preview",
            "quality_metrics",
            "iiif",
            "catalog_export",
            "search_index",
            "clustering",
        ],
    }

    Path(config_path).write_text(yaml.dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote configuration to {config_path}")


if __name__ == "__main__":
    main()

