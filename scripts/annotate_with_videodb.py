#!/usr/bin/env python3
"""
annotate_with_videodb.py

Automated autonomous driving video scenario annotation using VideoDB SDK.
Uploads driving videos, runs multi-analyzer perception (VLM + Object Detection)
grounded in the scenario taxonomy, and generates schema-compliant annotation JSON files.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Load .env file
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

try:
    import videodb
except ImportError:
    print("Error: 'videodb' is not installed in the active environment.")
    print("Please run: ./venv/bin/pip install 'videodb>=0.5.0' python-dotenv jsonschema")
    sys.exit(1)

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

VIDEOS_DIR = BASE_DIR / "videos"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
SCHEMA_PATH = BASE_DIR / "schema" / "annotation_schema.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

TAXONOMY_PROMPT = """
You are an expert autonomous driving scenario annotator. Analyze this video recording from an ego vehicle perspective and annotate the primary event, environment, and interacting actors according to the following strict taxonomy:

1. EVENT TYPE:
- 'Traffic Control': Traffic light changes, traffic sign recognition (speed limits, stop signs, yield, one-way, no entry).
- 'Vehicle Interaction': Cut-ins, cut-outs, lane changes, sudden stops from leading vehicles, vehicle overtaking, merging.
- 'Road & Environment': Changes in road conditions (potholes, speed bumps, wet road, debris), sudden weather shifts affecting visibility (rain, fog).
- 'Obstacle Detection': Moving obstacles (vehicles, bicycles, pedestrians, animals) or stationary objects (parked cars, cones, fallen trees, barriers).
- 'Animal & Pedestrian': Animals crossing/on the road, pedestrians on sidewalks or crossing/entering roadways.
- 'Sudden & Unexpected': Emergency vehicles (ambulance, fire truck), crashes, breakdowns, erratic driving.

2. EVENT PATTERN:
- 'Co-linear': Events on the same route/lane as ego vehicle (in line with ego vehicle travel).
- 'Cross-linear': Events crossing the ego vehicle route perpendicularly or from sides/intersections.
- 'Causal': Direct trigger-action cause-effect (e.g. lead car brakes, ego brakes).
- 'Likelihood': Anticipating potential future hazard (e.g. ball rolls onto street -> likely child follows).

3. ENVIRONMENT:
- road_type: 'Urban', 'Semi-urban', 'Rural', 'Highway', or 'Unknown'
- intersection: boolean (true if an intersection/crossroad is present, false otherwise)
- traffic_control: 'Traffic signal', 'Stop sign', 'Yield', 'None', or 'Unknown'
- weather: 'Clear', 'Rain', 'Fog', or 'Cloudy'
- lighting: 'Day', 'Night', or 'Dawn/Dusk'

4. SCENARIO BEHAVIOR:
- severity: 'Low', 'Medium', 'Moderate', or 'High'
- ego_maneuver: 'Go straight', 'Left Turn', 'Right Turn', 'Lane change', 'Cut-in', 'Cut-out', 'Overtake', 'Emergency brake', 'Yield', 'Follow', 'Stop', or 'Other'
- key_event: Concise one-line description of the critical event (e.g., 'Vehicle enters ego lane from right').
- time_window: start_sec and end_sec in 'MM:SS' format (e.g. '00:05' to '00:12') representing when the event occurs.

5. ACTORS:
List all interacting or notable actors with:
- class: e.g. 'car', 'pedestrian', 'motorcycle', 'bicycle', 'truck', 'bus', 'animal', 'auto_rickshaw'
- count: integer count
- relative_position: position relative to ego vehicle, e.g. 'front_right', 'front_left', 'front', 'ego_lane', 'sidewalk_left', 'sidewalk_right', 'crossing', 'rear'

Return the result conforming precisely to the requested schema.
"""

VLM_SCHEMA = {
    "environment": {
        "road_type": {
            "type": "enum",
            "values": ["Urban", "Semi-urban", "Rural", "Highway", "Unknown"]
        },
        "intersection": "boolean",
        "traffic_control": {
            "type": "enum",
            "values": ["Traffic signal", "Stop sign", "Yield", "None", "Unknown"]
        },
        "weather": {
            "type": "enum",
            "values": ["Clear", "Rain", "Fog", "Cloudy"]
        },
        "lighting": {
            "type": "enum",
            "values": ["Day", "Night", "Dawn/Dusk"]
        }
    },
    "scenario_behavior": {
        "event_type": {
            "type": "enum",
            "values": [
                "Traffic Control",
                "Vehicle Interaction",
                "Road & Environment",
                "Obstacle Detection",
                "Animal & Pedestrian",
                "Sudden & Unexpected"
            ]
        },
        "event_pattern": {
            "type": "enum",
            "values": ["Co-linear", "Cross-linear", "Causal", "Likelihood"]
        },
        "severity": {
            "type": "enum",
            "values": ["Low", "Medium", "Moderate", "High"]
        },
        "ego_maneuver": {
            "type": "enum",
            "values": [
                "Go straight",
                "Left Turn",
                "Right Turn",
                "Lane change",
                "Cut-in",
                "Cut-out",
                "Overtake",
                "Emergency brake",
                "Yield",
                "Follow",
                "Stop",
                "Other"
            ]
        },
        "key_event": "text",
        "time_window": {
            "start_sec": "string",
            "end_sec": "string"
        }
    },
    "actors": [
        {
            "class": "string",
            "count": "integer",
            "relative_position": "string"
        }
    ]
}


def verify_api_key():
    api_key = os.getenv("VIDEO_DB_API_KEY", "").strip()
    if not api_key or api_key == "your_videodb_api_key_here":
        print("\n" + "=" * 60)
        print("ERROR: VIDEO_DB_API_KEY is not configured!")
        print("=" * 60)
        print("1. Open the file: /home/isha/Project/Annotation/.env")
        print("2. Replace 'your_videodb_api_key_here' with your real API key.")
        print("   (Obtain a free key at https://console.videodb.io)")
        print("3. Re-run this command.")
        print("=" * 60 + "\n")
        sys.exit(1)
    return api_key


def load_validation_schema():
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None



VALID_ROAD_TYPES = ["Urban", "Semi-urban", "Rural", "Highway", "Unknown"]
VALID_TRAFFIC_CONTROLS = ["Traffic signal", "Stop sign", "Yield", "None", "Unknown"]
VALID_WEATHERS = ["Clear", "Rain", "Fog", "Cloudy"]
VALID_LIGHTINGS = ["Day", "Night", "Dawn/Dusk"]
VALID_EVENT_TYPES = [
    "Traffic Control",
    "Vehicle Interaction",
    "Road & Environment",
    "Obstacle Detection",
    "Animal & Pedestrian",
    "Sudden & Unexpected"
]
VALID_EVENT_PATTERNS = ["Co-linear", "Cross-linear", "Causal", "Likelihood"]
VALID_SEVERITIES = ["Low", "Medium", "Moderate", "High"]
VALID_MANEUVERS = [
    "Go straight",
    "Left Turn",
    "Right Turn",
    "Lane change",
    "Cut-in",
    "Cut-out",
    "Overtake",
    "Emergency brake",
    "Yield",
    "Follow",
    "Stop",
    "Other"
]


def normalize_choice(val, valid_choices, default):
    if not val or not isinstance(val, str):
        return default
    val_clean = val.strip()
    for choice in valid_choices:
        if val_clean.lower() == choice.lower():
            return choice
    for choice in valid_choices:
        if val_clean.lower() in choice.lower() or choice.lower() in val_clean.lower():
            return choice
    return default


def annotate_video(video_path: Path, conn, model_tier="pro", segment_seconds=5, overwrite=False):
    video_id = video_path.stem
    try:
        rel_dir = video_path.resolve().relative_to(VIDEOS_DIR.resolve()).parent
    except ValueError:
        rel_dir = Path(".")
    if rel_dir.name.lower() == "idd":
        rel_dir = Path("IDD")
    output_json_path = ANNOTATIONS_DIR / rel_dir / f"{video_id}.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if output_json_path.exists() and not overwrite:
        print(f"Skipping {video_id}: Annotation file already exists ({output_json_path}). Use --overwrite to re-run.")
        return

    try:
        print(f"\n[{video_id}] Uploading {video_path.name} to VideoDB...")
        coll = conn.get_collection()
        uploaded_video = coll.upload(file_path=str(video_path))
        print(f"[{video_id}] Upload complete. Video ID: {uploaded_video.id}")

        player_url = f"https://console.videodb.io/player?url={uploaded_video.stream_url}" if hasattr(uploaded_video, "stream_url") else ""

        print(f"[{video_id}] Running VLM understanding & scene analysis (model: {model_tier})...")
        understanding = uploaded_video.understand(
            analyzers=[
                {
                    "type": "vlm",
                    "name": "scenario_annotation",
                    "config": {
                        "model": model_tier,
                        "prompt": TAXONOMY_PROMPT,
                        "schema": VLM_SCHEMA
                    }
                }
            ],
            segmentation={"type": "time", "seconds": segment_seconds}
        )

        print(f"[{video_id}] Processing analyzers (polling for completion)...")
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                understanding.refresh()
                if understanding.is_complete:
                    break
            except Exception:
                # VideoDB server can have transient 500/connection blips while processing
                pass
            time.sleep(5)
            print(".", end="", flush=True)
        print(" Completed!")

        # Extract scenario annotation artifact
        output = understanding.get_analyzer_output("scenario_annotation")
        
        # Parse output structure
        env_data = {}
        behavior_data = {}
        actors_data = []

        if isinstance(output, str):
            try:
                cleaned_str = output.strip()
                if cleaned_str.startswith("```json"):
                    cleaned_str = cleaned_str[7:]
                if cleaned_str.endswith("```"):
                    cleaned_str = cleaned_str[:-3]
                output = json.loads(cleaned_str.strip())
            except Exception:
                pass

        # Extract scenes if available
        scenes_list = []
        if isinstance(output, dict):
            if "scenes" in output and isinstance(output["scenes"], list):
                scenes_list = output["scenes"]
            elif "environment" in output or "scenario_behavior" in output:
                scenes_list = [{"data": output}]
        elif isinstance(output, list):
            scenes_list = output

        # Pick the most prominent scene (prioritize high/medium severity or active interaction)
        chosen_scene_data = {}
        severity_rank = {"High": 3, "Medium": 2, "Moderate": 2, "Low": 1}
        best_rank = 0

        for s in scenes_list:
            sdata = s.get("data", s) if isinstance(s, dict) else {}
            if not isinstance(sdata, dict):
                continue
            behav = sdata.get("scenario_behavior", {})
            sev = behav.get("severity", "Low")
            rank = severity_rank.get(sev, 1)

            # Bonus rank if ego is yielding, turning, changing lane, or braking
            maneuver = behav.get("ego_maneuver", "Go straight")
            if maneuver not in ("Go straight", "Follow"):
                rank += 1

            if rank > best_rank or not chosen_scene_data:
                best_rank = rank
                chosen_scene_data = sdata

        if chosen_scene_data:
            env_data = chosen_scene_data.get("environment", {})
            behavior_data = chosen_scene_data.get("scenario_behavior", {})
            actors_data = chosen_scene_data.get("actors", [])

        # Smart dataset detection
        if "dashcam" in str(video_path).lower() or video_id.startswith("EVT_") or video_id.startswith("REC_"):
            dataset_name = "Dashcam"
        elif video_id.startswith("scene-") or video_id.startswith("nusc"):
            dataset_name = "NuScenes"
        elif video_id.startswith("2011_") or video_id.startswith("kitti"):
            dataset_name = "KITTI"
        elif "idd" in str(video_path).lower() or "combined" in video_id.lower():
            dataset_name = "IDD"
        else:
            dataset_name = video_id.split("_")[0].upper()

        # Normalize time_window
        tw = behavior_data.get("time_window", {})
        if not isinstance(tw, dict):
            tw = {"start_sec": "00:00", "end_sec": "00:05"}
        if "start_sec" not in tw:
            tw["start_sec"] = "00:00"
        if "end_sec" not in tw:
            tw["end_sec"] = "00:05"
        tw["start_sec"] = str(tw["start_sec"])
        tw["end_sec"] = str(tw["end_sec"])

        # Clean and validate actors
        cleaned_actors = []
        if isinstance(actors_data, list):
            for act in actors_data:
                if isinstance(act, dict) and "class" in act:
                    cnt = act.get("count", 1)
                    try:
                        cnt = int(cnt)
                        if cnt < 1:
                            cnt = 1
                    except (ValueError, TypeError):
                        cnt = 1
                    pos = act.get("relative_position", "front")
                    # Clean position string - take first if comma separated
                    pos_str = str(pos).lower().strip().split(",")[0].strip()
                    pos_str = pos_str.replace(" ", "_").replace("-", "_")
                    if not pos_str:
                        pos_str = "front"
                    cleaned_actors.append({
                        "class": str(act["class"]).lower(),
                        "count": cnt,
                        "relative_position": pos_str
                    })

        if not cleaned_actors:
            cleaned_actors = [
                {"class": "car", "count": 1, "relative_position": "front"}
            ]

        # Strictly stick to user's specified JSON format (omit any VideoDB internal IDs)
        final_annotation = {
            "metadata": {
                "video_id": video_id,
                "dataset": dataset_name
            },
            "environment": {
                "road_type": normalize_choice(env_data.get("road_type"), VALID_ROAD_TYPES, "Urban"),
                "intersection": bool(env_data.get("intersection", False)),
                "traffic_control": normalize_choice(env_data.get("traffic_control"), VALID_TRAFFIC_CONTROLS, "None"),
                "weather": normalize_choice(env_data.get("weather"), VALID_WEATHERS, "Clear"),
                "lighting": normalize_choice(env_data.get("lighting"), VALID_LIGHTINGS, "Day")
            },
            "scenario_behavior": {
                "event_type": normalize_choice(behavior_data.get("event_type"), VALID_EVENT_TYPES, "Vehicle Interaction"),
                "event_pattern": normalize_choice(behavior_data.get("event_pattern"), VALID_EVENT_PATTERNS, "Co-linear"),
                "severity": normalize_choice(behavior_data.get("severity"), VALID_SEVERITIES, "Low"),
                "ego_maneuver": normalize_choice(behavior_data.get("ego_maneuver"), VALID_MANEUVERS, "Go straight"),
                "key_event": str(behavior_data.get("key_event", f"Driving interaction in {video_id}")).strip() or f"Driving interaction in {video_id}",
                "time_window": tw
            },
            "actors": cleaned_actors
        }

        # Validate against project schema
        val_schema = load_validation_schema()
        if val_schema and HAS_JSONSCHEMA:
            try:
                jsonschema.validate(instance=final_annotation, schema=val_schema)
                print(f"[{video_id}] Schema Validation: PASSED ✓")
            except jsonschema.ValidationError as err:
                print(f"[{video_id}] Schema Validation Warning: {err.message}")

        # Write output JSON
        ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_annotation, f, indent=2)
            f.write("\n")

        print(f"[{video_id}] Saved annotation to: {output_json_path}")
        if player_url:
            print(f"[{video_id}] Playable video stream link: {player_url}")
    except Exception as err:
        print(f"[{video_id}] ERROR processing video: {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate autonomous driving videos using VideoDB VLM perception."
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Specific video filename or path (e.g., IDD_0234.mp4 or videos/IDD_0234.mp4)"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="List of specific video filenames or paths to annotate."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Annotate all videos currently in the videos/ folder."
    )
    parser.add_argument(
        "--nuscenes",
        action="store_true",
        help="Annotate all NuScenes videos in videos/nuscenes dataset/ or starting with 'scene-'"
    )
    parser.add_argument(
        "--kitti",
        action="store_true",
        help="Annotate all KITTI videos in videos/kitti dataset/ or starting with '2011_'"
    )
    parser.add_argument(
        "--dashcam",
        action="store_true",
        help="Annotate all Dashcam videos in videos/dashcam/"
    )
    parser.add_argument(
        "--idd",
        action="store_true",
        help="Annotate all IDD videos in videos/IDD/, videos/idd/ or starting with 'idd_'"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit maximum number of videos to annotate in this run."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation JSON files."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pro",
        choices=["mini", "basic", "pro", "ultra"],
        help="VideoDB model tier to use for VLM understanding (default: 'pro')."
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=5,
        help="Segmentation window in seconds (default: 5)."
    )

    args = parser.parse_args()

    # Ensure API key is configured
    verify_api_key()

    # Connect to VideoDB
    print("Connecting to VideoDB...")
    conn = videodb.connect()

    target_videos = []
    if args.video:
        vpath = Path(args.video)
        if not vpath.is_file():
            vpath = VIDEOS_DIR / args.video
        if not vpath.is_file():
            print(f"Error: Video file '{args.video}' not found in current directory or videos/.")
            sys.exit(1)
        target_videos.append(vpath)
    elif args.videos:
        for vid in args.videos:
            vpath = Path(vid)
            if not vpath.is_file():
                vpath = VIDEOS_DIR / vid
            if not vpath.is_file():
                print(f"Error: Video file '{vid}' not found in current directory or videos/.")
                sys.exit(1)
            target_videos.append(vpath)
    elif args.nuscenes:
        candidates = list((VIDEOS_DIR / "nuscenes dataset").rglob("*.mp4")) + list(VIDEOS_DIR.glob("scene-*.mp4"))
        seen = set()
        for f in sorted(candidates):
            rf = f.resolve()
            if rf not in seen and f.is_file():
                seen.add(rf)
                target_videos.append(f)
        if not target_videos:
            print("No NuScenes video files found in videos/nuscenes dataset/ or videos/.")
            sys.exit(0)
    elif args.kitti:
        candidates = list((VIDEOS_DIR / "kitti dataset").rglob("*.mp4")) + list(VIDEOS_DIR.glob("2011_*.mp4"))
        seen = set()
        for f in sorted(candidates):
            rf = f.resolve()
            if rf not in seen and f.is_file():
                seen.add(rf)
                target_videos.append(f)
        if not target_videos:
            print("No KITTI video files found in videos/kitti dataset/ or videos/.")
            sys.exit(0)
    elif args.dashcam:
        candidates = list((VIDEOS_DIR / "dashcam").rglob("*.mp4"))
        seen = set()
        for f in sorted(candidates):
            rf = f.resolve()
            if rf not in seen and f.is_file():
                seen.add(rf)
                target_videos.append(f)
        if not target_videos:
            print("No Dashcam video files found in videos/dashcam/.")
            sys.exit(0)
    elif args.idd:
        candidates = (
            list((VIDEOS_DIR / "IDD").rglob("*.mp4")) +
            list((VIDEOS_DIR / "idd").rglob("*.mp4")) +
            list(VIDEOS_DIR.glob("idd_*.mp4"))
        )
        seen = set()
        for f in sorted(candidates):
            rf = f.resolve()
            if rf not in seen and f.is_file():
                seen.add(rf)
                target_videos.append(f)
        if not target_videos:
            print("No IDD video files found in videos/IDD/, videos/idd/, or videos/.")
            sys.exit(0)
    elif args.all:
        target_videos = [
            f for f in sorted(VIDEOS_DIR.iterdir())
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not target_videos:
            print(f"No video files found in {VIDEOS_DIR}. Please place your videos in videos/ first.")
            sys.exit(0)
    else:
        # Default: check videos in videos/
        target_videos = [
            f for f in sorted(VIDEOS_DIR.iterdir())
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not target_videos:
            print("No video specified. Usage:")
            print("  ./venv/bin/python scripts/annotate_with_videodb.py --video videos/IDD_0234.mp4")
            print("  ./venv/bin/python scripts/annotate_with_videodb.py --nuscenes")
            print("  ./venv/bin/python scripts/annotate_with_videodb.py --kitti")
            print("  ./venv/bin/python scripts/annotate_with_videodb.py --dashcam")
            print("  ./venv/bin/python scripts/annotate_with_videodb.py --idd")
            print("  ./venv/bin/python scripts/annotate_with_videodb.py --all")
            sys.exit(0)

    # Filter already annotated if not overwriting
    if not args.overwrite:
        unannotated = []
        for v in target_videos:
            try:
                rel_dir = v.resolve().relative_to(VIDEOS_DIR.resolve()).parent
            except ValueError:
                rel_dir = Path(".")
            if rel_dir.name.lower() == "idd":
                rel_dir = Path("IDD")
            out_p = ANNOTATIONS_DIR / rel_dir / f"{v.stem}.json"
            if not out_p.exists():
                unannotated.append(v)
        print(f"Total candidates: {len(target_videos)} | Unannotated: {len(unannotated)} | Already annotated: {len(target_videos) - len(unannotated)}")
        target_videos = unannotated

    if args.limit and args.limit > 0:
        target_videos = target_videos[:args.limit]
        print(f"Limiting to first {len(target_videos)} video(s) for this batch.")

    if not target_videos:
        print("All target videos are already annotated!")
        sys.exit(0)

    print(f"Processing {len(target_videos)} video(s) in this run.")
    for idx, v in enumerate(target_videos, start=1):
        print(f"\n[{idx}/{len(target_videos)}] Target: {v.name}")
        annotate_video(
            v,
            conn,
            model_tier=args.model,
            segment_seconds=args.segment_seconds,
            overwrite=args.overwrite
        )

    print("\nAnnotation run complete!")


if __name__ == "__main__":
    main()
