import argparse
import json
import os
import shutil
import sys
import traceback
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, Template

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
    preview_cfg = cfg.setdefault("preview_dashboard", {})
    if not preview_cfg.get("enabled", False):
        raise ValueError(
            "Preview dashboard is disabled in the configuration. "
            "Set `preview_dashboard.enabled` to true.",
        )

    outputs_cfg = cfg.setdefault("outputs", {})
    base_dir = outputs_cfg.get("base_dir", "transcripts")
    outputs_cfg["base_dir"] = expand_path(base_dir)

    format_dirs = {}
    for fmt in outputs_cfg.get("formats", []):
        format_dirs[fmt] = os.path.join(outputs_cfg["base_dir"], fmt)
    outputs_cfg["format_dirs"] = {fmt: expand_path(path) for fmt, path in format_dirs.items()}

    preview_cfg["output_path"] = expand_path(preview_cfg.get("output_path", "preview/index.html"))
    preview_cfg["static_assets_dir"] = expand_path(preview_cfg.get("static_assets_dir", "preview/assets"))

    return cfg


def read_text_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_json(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def gather_items(cfg: Dict) -> List[Dict]:
    preview_cfg = cfg["preview_dashboard"]
    outputs_cfg = cfg.get("outputs", {})
    format_dirs = outputs_cfg.get("format_dirs", {})

    transcript_dir = format_dirs.get("txt")
    summary_dir = cfg.get("summarization", {}).get("output_dir", "summaries")
    summary_dir = expand_path(summary_dir)
    tags_dir = cfg.get("tagging", {}).get("output_dir", "tags")
    tags_dir = expand_path(tags_dir)

    frame_root = cfg.get("frame_descriptions", {}).get("output_dir", "frame_descriptions")
    frame_root = expand_path(frame_root)
    keyframe_root = cfg.get("keyframes", {}).get("output_dir", "keyframes")
    keyframe_root = expand_path(keyframe_root)

    ids = set()
    if transcript_dir and os.path.exists(transcript_dir):
        ids.update(os.path.splitext(f)[0] for f in os.listdir(transcript_dir))
    if os.path.exists(summary_dir):
        ids.update(os.path.splitext(f)[0] for f in os.listdir(summary_dir))
    if os.path.exists(tags_dir):
        ids.update(os.path.splitext(f)[0] for f in os.listdir(tags_dir))

    items = []
    max_items = preview_cfg.get("max_items")
    for idx, item_id in enumerate(sorted(ids)):
        if max_items is not None and idx >= max_items:
            break
        transcript = read_text_file(
            os.path.join(transcript_dir, f"{item_id}.txt") if transcript_dir else None,
        )
        summary = read_text_file(os.path.join(summary_dir, f"{item_id}.txt"))
        tags = load_json(os.path.join(tags_dir, f"{item_id}.json"))

        frames = []
        if preview_cfg.get("include_frames", True) and os.path.exists(frame_root):
            pattern_dir = None
            for root, _, files in os.walk(frame_root):
                matching = [f for f in files if f.startswith(f"{item_id}_t")]
                if matching:
                    pattern_dir = root
                    for fname in sorted(matching)[:6]:
                        rel = os.path.relpath(os.path.join(root, fname), frame_root)
                        with open(os.path.join(root, fname), "r", encoding="utf-8") as handle:
                            description = handle.read().strip()
                        timestamp = fname.split("_t")[-1].split(".")[0]
                        frame_image = find_corresponding_image(keyframe_root, rel)
                        frames.append(
                            {
                                "description": description,
                                "timestamp": timestamp,
                                "image": frame_image,
                            },
                        )
                    break

        item = {
            "id": item_id,
            "transcript": transcript if preview_cfg.get("include_transcript", False) else None,
            "summary": summary if preview_cfg.get("include_summaries", True) else None,
            "tags": tags if preview_cfg.get("include_tags", True) else None,
            "metadata": None,
            "frames": frames,
        }
        items.append(item)
    return items


def find_corresponding_image(keyframe_root: str, description_rel_path: str) -> Optional[str]:
    if not os.path.exists(keyframe_root):
        return None
    base_name = os.path.splitext(os.path.basename(description_rel_path))[0] + ".jpg"
    for mode_dir in os.listdir(keyframe_root):
        candidate = os.path.join(keyframe_root, mode_dir, base_name)
        if os.path.exists(candidate):
            return os.path.relpath(candidate, keyframe_root)
    return None


def render_template(items: List[Dict], cfg: Dict) -> str:
    preview_cfg = cfg["preview_dashboard"]
    template_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Collection Preview</title>
        <link rel="stylesheet" href="{{ css_href }}">
    </head>
    <body>
    <header>
        <h1>Collection Preview</h1>
        <p>Total items previewed: {{ items|length }}</p>
    </header>
    {% for item in items %}
        <section class="item">
            <h2>{{ item.id }}</h2>
            {% if item.summary %}
            <article class="summary">
                <h3>Summary</h3>
                <p>{{ item.summary }}</p>
            </article>
            {% endif %}
            {% if item.tags %}
            <article class="tags">
                <h3>Tags</h3>
                <pre>{{ item.tags | tojson(indent=2) }}</pre>
            </article>
            {% endif %}
            {% if item.frames %}
            <article class="frames">
                <h3>Sample Frames</h3>
                <ul>
                {% for frame in item.frames %}
                    <li>
                        {% if frame.image %}
                            <figure>
                                <img src="{{ frames_base }}/{{ frame.image }}" alt="Frame {{ frame.timestamp }}" />
                                <figcaption>{{ frame.timestamp }} ms — {{ frame.description }}</figcaption>
                            </figure>
                        {% else %}
                            <strong>{{ frame.timestamp }} ms</strong>: {{ frame.description }}
                        {% endif %}
                    </li>
                {% endfor %}
                </ul>
            </article>
            {% endif %}
            {% if item.transcript %}
            <details class="transcript">
                <summary>Transcript</summary>
                <pre>{{ item.transcript }}</pre>
            </details>
            {% endif %}
        </section>
    {% endfor %}
    </body>
    </html>
    """
    env = Environment()
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
    template: Template = env.from_string(template_html)
    css_href = preview_cfg.get("css_file")
    if css_href:
        css_href = os.path.relpath(css_href, os.path.dirname(preview_cfg["output_path"]))
    else:
        css_href = "assets/style.css"
    frames_base = os.path.relpath(
        cfg.get("keyframes", {}).get("output_dir", "keyframes"),
        os.path.dirname(preview_cfg["output_path"]),
    )
    return template.render(items=items, css_href=css_href, frames_base=frames_base)


def write_assets(preview_cfg: Dict) -> None:
    assets_dir = preview_cfg["static_assets_dir"]
    os.makedirs(assets_dir, exist_ok=True)
    css_path = preview_cfg.get("css_file")
    if css_path and os.path.exists(css_path):
        shutil.copy(css_path, os.path.join(assets_dir, "style.css"))
        return
    # Write default CSS
    default_css = """
    body {
        font-family: system-ui, sans-serif;
        margin: 2rem;
        background: #f7f7f7;
    }
    header {
        border-bottom: 1px solid #ccc;
        margin-bottom: 2rem;
    }
    section.item {
        background: white;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    article {
        margin-bottom: 1rem;
    }
    article.frames ul {
        list-style: none;
        padding: 0;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 1rem;
    }
    article.frames figure {
        margin: 0;
    }
    article.frames img {
        max-width: 100%;
        border-radius: 6px;
        border: 1px solid #ddd;
    }
    details.transcript pre {
        max-height: 240px;
        overflow-y: auto;
        background: #fafafa;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #eee;
    }
    """
    with open(os.path.join(assets_dir, "style.css"), "w", encoding="utf-8") as handle:
        handle.write(default_css)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a static HTML preview dashboard for the processed collection.",
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

    preview_cfg = config["preview_dashboard"]
    items = gather_items(config)

    html = render_template(items, config)

    os.makedirs(os.path.dirname(preview_cfg["output_path"]), exist_ok=True)
    with open(preview_cfg["output_path"], "w", encoding="utf-8") as handle:
        handle.write(html)

    write_assets(preview_cfg)

    print(f"Wrote preview dashboard to {preview_cfg['output_path']}")


if __name__ == "__main__":
    main()

