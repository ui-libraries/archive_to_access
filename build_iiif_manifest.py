import argparse
import json
import os
import sys
import traceback
from typing import Dict, List

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
    iiif_cfg = cfg.setdefault("iiif", {})
    if not iiif_cfg.get("enabled", False):
        raise ValueError(
            "IIIF manifest generation is disabled in the configuration. "
            "Set `iiif.enabled` to true.",
        )

    frames_root = iiif_cfg.get("frames_root")
    if frames_root:
        iiif_cfg["frames_root"] = expand_path(frames_root)
    else:
        key_cfg = cfg.get("keyframes", {})
        if key_cfg.get("enabled"):
            iiif_cfg["frames_root"] = expand_path(key_cfg.get("output_dir", "keyframes"))
        else:
            raise ValueError("`iiif.frames_root` must be set when keyframes are not enabled.")

    iiif_cfg["manifest_output"] = expand_path(iiif_cfg.get("manifest_output", "iiif/manifest.json"))

    metadata_csv = cfg.get("tagging", {}).get("metadata_csv") or cfg.get("summarization", {}).get("metadata_csv")
    if metadata_csv:
        iiif_cfg["metadata_csv"] = expand_path(metadata_csv)

    return cfg


def discover_items(frames_root: str) -> Dict[str, List[str]]:
    items: Dict[str, List[str]] = {}
    for mode in os.listdir(frames_root):
        mode_dir = os.path.join(frames_root, mode)
        if not os.path.isdir(mode_dir):
            continue
        for fname in sorted(os.listdir(mode_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            item_id = fname.split("_t")[0]
            rel_path = os.path.join(mode, fname)
            items.setdefault(item_id, []).append(rel_path)
    return items


def load_metadata(iiif_cfg: Dict) -> Dict[str, Dict[str, str]]:
    metadata_path = iiif_cfg.get("metadata_csv")
    if not metadata_path or not os.path.exists(metadata_path):
        return {}
    import csv

    metadata: Dict[str, Dict[str, str]] = {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identifier = row.get("ID") or row.get("id")
            if not identifier:
                continue
            metadata[identifier] = row
    return metadata


def build_manifest(items: Dict[str, List[str]], cfg: Dict) -> Dict:
    iiif_cfg = cfg["iiif"]
    metadata = load_metadata(iiif_cfg)
    frames_root = iiif_cfg["frames_root"]
    base_media_url = iiif_cfg.get("base_media_url", "https://example.org/media")
    base_frame_url = iiif_cfg.get("base_frame_url", "https://example.org/frames")
    manifest = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": f"{base_media_url}/manifest.json",
        "type": "Manifest",
        "label": {"en": ["AV Collection"]},
        "items": [],
    }

    for item_id, frames in sorted(items.items()):
        metadata_entries = metadata.get(item_id, {})
        metadata_fields = iiif_cfg.get("metadata_fields", [])
        metadata_labels = [
            {"label": {"en": [field]}, "value": {"en": [metadata_entries.get(field, "")]}}
            for field in metadata_fields
            if field in metadata_entries and metadata_entries[field]
        ]
        canvas = {
            "id": f"{base_media_url}/{item_id}/canvas",
            "type": "Canvas",
            "label": {"en": [item_id]},
            "items": [],
            "metadata": metadata_labels,
        }
        annotations = []
        for idx, frame_rel in enumerate(frames, start=1):
            frame_url = f"{base_frame_url}/{frame_rel}"
            annotation = {
                "id": f"{base_media_url}/{item_id}/annotation/{idx}",
                "type": "Annotation",
                "motivation": "painting",
                "body": {
                    "id": frame_url,
                    "type": "Image",
                    "format": "image/jpeg",
                },
                "target": f"{base_media_url}/{item_id}/canvas",
            }
            annotations.append(annotation)
        canvas["items"].append(
            {
                "id": f"{base_media_url}/{item_id}/page/1",
                "type": "AnnotationPage",
                "items": annotations,
            },
        )
        manifest["items"].append(canvas)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an IIIF manifest linking to generated keyframes.",
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

    iiif_cfg = config["iiif"]
    items = discover_items(iiif_cfg["frames_root"])
    if not items:
        sys.exit(f"No keyframes found under {iiif_cfg['frames_root']}.")

    manifest = build_manifest(items, config)
    os.makedirs(os.path.dirname(iiif_cfg["manifest_output"]), exist_ok=True)
    with open(iiif_cfg["manifest_output"], "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Wrote IIIF manifest to {iiif_cfg['manifest_output']}")


if __name__ == "__main__":
    main()

