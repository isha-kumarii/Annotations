#!/usr/bin/env python3
"""
build_dataset_json.py

Phase 1: Consolidates all individual scenario annotations into a single unified
JSON dataset catalog (dataset_catalog.json) optimized for dual-scoring hybrid retrieval.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ANNOTATIONS_DIR = BASE_DIR / "annotations_v2"
DEFAULT_OUTPUT_PATH = BASE_DIR / "dataset_catalog.json"


def construct_search_document(ann: dict) -> str:
    """
    Constructs a rich textual representation of the scenario for embedding/semantic search.
    """
    rn = ann.get("road_network", {})
    env = ann.get("environmental_conditions", {})
    ego = ann.get("ego_vehicle", {})
    actors = ann.get("npc_actors", [])
    evt = ann.get("event_details", {})

    actor_descs = [f"{a.get('actor_type', 'actor')} ({a.get('maneuver', 'moving')})" for a in actors]
    actor_str = ", ".join(actor_descs) if actor_descs else "none"

    signs = rn.get("traffic_sign", [])
    sign_str = f", signs: {', '.join(signs)}" if signs else ""

    doc = (
        f"Event: {evt.get('event_description', '')}. "
        f"Event Class: {evt.get('event_class', '')} ({evt.get('event_pattern', '')} pattern, severity: {evt.get('event_severity', '')}). "
        f"Road Network: {rn.get('road_type', '')} {rn.get('road_layout', '')} with {rn.get('no_of_lanes', '')} lanes, "
        f"intersection control: {rn.get('intersection_control', '')}{sign_str}, traffic light: {rn.get('traffic_light', 'none')}. "
        f"Environment: Weather {env.get('weather', '')}, Visibility {env.get('visibility', '')}, Lighting {env.get('lighting', '')}. "
        f"Ego Vehicle: maneuver '{ego.get('maneuver', '')}'. "
        f"NPC Actors: {actor_str}."
    )
    return doc.strip()


def build_catalog(annotations_dir: Path, output_file: Path) -> dict:
    if not annotations_dir.exists():
        print(f"Error: Annotations directory {annotations_dir} does not exist.")
        sys.exit(1)

    json_files = sorted(list(annotations_dir.rglob("*.json")))
    print(f"Scanning {annotations_dir}: Found {len(json_files)} annotation JSON files.")

    videos_catalog = []
    dataset_counts = {}

    for idx, jf in enumerate(json_files, start=1):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            meta = data.get("metadata", {})
            vid = meta.get("video_id", jf.stem)
            ds = meta.get("dataset", jf.parent.name)

            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

            # Build searchable representation
            search_doc = construct_search_document(data)
            actor_types = sorted(list(set(a.get("actor_type", "").lower() for a in data.get("npc_actors", []))))

            entry = {
                "video_id": vid,
                "dataset": ds,
                "video_path": meta.get("video_path", f"videos/{ds}/{vid}.mp4"),
                "annotation": {
                    "road_network": data.get("road_network", {}),
                    "environmental_conditions": data.get("environmental_conditions", {}),
                    "ego_vehicle": data.get("ego_vehicle", {}),
                    "npc_actors": data.get("npc_actors", []),
                    "event_details": data.get("event_details", {})
                },
                "search_index": {
                    "event_description": data.get("event_details", {}).get("event_description", ""),
                    "text_document": search_doc,
                    "actor_types": actor_types,
                    "event_class": data.get("event_details", {}).get("event_class", ""),
                    "weather": data.get("environmental_conditions", {}).get("weather", ""),
                    "road_type": data.get("road_network", {}).get("road_type", ""),
                    "ego_maneuver": data.get("ego_vehicle", {}).get("maneuver", "")
                }
            }
            videos_catalog.append(entry)

        except Exception as e:
            print(f"Warning: Failed to process {jf.name}: {e}")

    catalog = {
        "metadata": {
            "total_videos": len(videos_catalog),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(annotations_dir),
            "dataset_breakdown": dataset_counts
        },
        "videos": videos_catalog
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"\n✓ Successfully compiled {len(videos_catalog)} videos into: {output_file}")
    print("Dataset Breakdown:")
    for ds, cnt in dataset_counts.items():
        print(f"  - {ds}: {cnt} videos")

    return catalog


def main():
    parser = argparse.ArgumentParser(description="Compile all scenario annotations into unified dataset_catalog.json")
    parser.add_argument("--annotations-dir", type=str, default=str(DEFAULT_ANNOTATIONS_DIR), help="Directory containing JSON annotations.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Destination JSON file.")
    args = parser.parse_args()

    build_catalog(Path(args.annotations_dir), Path(args.output))


if __name__ == "__main__":
    main()

