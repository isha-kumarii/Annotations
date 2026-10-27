#!/usr/bin/env python3
"""
generate_templates.py

Scans the videos/ directory for video files (.mp4, .avi, .mkv, .mov, etc.)
and generates starter JSON annotation files in annotations/ for any videos
that do not already have an annotation file.
"""

from pathlib import Path
import json

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
TEMPLATE_PATH = BASE_DIR / "schema" / "template.json"


def load_template():
    if TEMPLATE_PATH.exists():
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "metadata": {
            "video_id": "",
            "dataset": "IDD"
        },
        "environment": {
            "road_type": "Urban",
            "intersection": False,
            "traffic_control": "None",
            "weather": "Clear",
            "lighting": "Day"
        },
        "scenario_behavior": {
            "event_type": "Vehicle Interaction",
            "event_pattern": "Co-linear",
            "severity": "Low",
            "ego_maneuver": "Go straight",
            "key_event": "",
            "time_window": {
                "start_sec": "00:00",
                "end_sec": "00:00"
            }
        },
        "actors": []
    }


def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in VIDEOS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No video files found in {VIDEOS_DIR}.")
        print("Please paste your video files (.mp4, .avi, etc.) into the 'videos/' directory.")
        return

    template = load_template()
    created_count = 0

    for video_file in sorted(video_files):
        video_id = video_file.stem
        annotation_file = ANNOTATIONS_DIR / f"{video_id}.json"

        if annotation_file.exists():
            print(f"[EXISTS] {annotation_file.name}")
            continue

        dataset_name = video_id.split("_")[0] if "_" in video_id else "IDD"
        data = json.loads(json.dumps(template))  # Deep copy
        data["metadata"]["video_id"] = video_id
        data["metadata"]["dataset"] = dataset_name

        with open(annotation_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

        print(f"[CREATED] {annotation_file.name}")
        created_count += 1

    print(f"\nDone! Generated {created_count} new annotation template(s).")


if __name__ == "__main__":
    main()

