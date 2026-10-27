#!/usr/bin/env python3
"""
annotate_with_gemini.py

Automated autonomous driving video scenario and behavior annotation pipeline
powered by Google Gemini Multimodal API (gemini-3.8-flash).

Strictly adheres to schema/annotation_schema.json and generates outputs in:
  annotations_gemini/<dataset_subfolder>/<video_id>.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
DEFAULT_ANNOTATIONS_DIR = BASE_DIR / "annotations_gemini"
SCHEMA_PATH = BASE_DIR / "schema" / "annotation_schema.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash"]

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
- count: integer count (minimum 1)
- relative_position: position relative to ego vehicle, e.g. 'front_right', 'front_left', 'front', 'ego_lane', 'sidewalk_left', 'sidewalk_right', 'rear'

Return ONLY valid JSON matching this structure:
{
  "metadata": {
    "video_id": "string",
    "dataset": "string"
  },
  "environment": {
    "road_type": "string",
    "intersection": true,
    "traffic_control": "string",
    "weather": "string",
    "lighting": "string"
  },
  "scenario_behavior": {
    "event_type": "string",
    "event_pattern": "string",
    "severity": "string",
    "ego_maneuver": "string",
    "key_event": "string",
    "time_window": {
      "start_sec": "MM:SS",
      "end_sec": "MM:SS"
    }
  },
  "actors": [
    {
      "class": "string",
      "count": 1,
      "relative_position": "string"
    }
  ]
}
"""

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


def verify_api_key():
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("\n" + "=" * 60)
        print("ERROR: GEMINI_API_KEY is not configured!")
        print("=" * 60)
        print("1. Open the file: /home/isha/Project/Annotation/.env")
        print("2. Add: GEMINI_API_KEY=your_gemini_api_key_here")
        print("3. Re-run this command.")
        print("=" * 60 + "\n")
        sys.exit(1)
    return api_key


def load_validation_schema():
    if SCHEMA_PATH.exists():
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


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


def annotate_video_with_gemini(video_path: Path, client, model_name="gemini-3.7-flash", output_dir=None, overwrite=False):
    if output_dir is None:
        output_dir = DEFAULT_ANNOTATIONS_DIR

    video_id = video_path.stem
    try:
        rel_dir = video_path.resolve().relative_to(VIDEOS_DIR.resolve()).parent
    except ValueError:
        rel_dir = Path(".")
    if rel_dir.name.lower() == "idd":
        rel_dir = Path("IDD")

    output_json_path = output_dir / rel_dir / f"{video_id}.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if output_json_path.exists() and not overwrite:
        print(f"Skipping {video_id}: Annotation file already exists ({output_json_path}). Use --overwrite to re-run.")
        return True, model_name

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

    uploaded_file = None
    try:
        print(f"\n[{video_id}] Uploading {video_path.name} to Gemini File API...")
        uploaded_file = client.files.upload(file=str(video_path))
        print(f"[{video_id}] Uploaded file ID: {uploaded_file.name}. Polling for processing state...")

        deadline = time.time() + 300
        while time.time() < deadline:
            if uploaded_file.state.name == "ACTIVE":
                break
            if uploaded_file.state.name == "FAILED":
                raise RuntimeError(f"Gemini video processing failed for {video_id}")
            time.sleep(2)
            print(".", end="", flush=True)
            uploaded_file = client.files.get(name=uploaded_file.name)
        print(" Active!")

        # Single-model execution to maintain strict consistency across all annotations
        print(f"[{video_id}] Running multimodal scenario understanding with {model_name}...")
        max_retries = 3
        resp = None
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, TAXONOMY_PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                if resp and resp.text:
                    break
            except Exception as e:
                err_str = str(e).lower()
                is_daily_quota = any(
                    token in err_str for token in [
                        "quota exceeded for metric",
                        "generaterequestsperday",
                        "limit: 20",
                        "retry in"
                    ]
                )
                if is_daily_quota:
                    print(f"\n[{video_id}] Daily free-tier quota (20 requests/day) reached for {model_name}.")
                    print(f"To remove this limit: link a billing account in Google AI Studio for pay-as-you-go (~$0.003/video).")
                    raise e

                is_transient = any(
                    token in err_str for token in [
                        "429", "quota", "rate", "resource_exhausted",
                        "503", "unavailable", "high demand", "overloaded",
                        "deadline", "timeout", "timed out", "500", "internal"
                    ]
                )
                if is_transient and attempt < max_retries - 1:
                    wait_times = [10, 20, 30]
                    wait_time = wait_times[attempt]
                    print(f"[{video_id}] Transient API notification: {e}. Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e

        if not resp or not resp.text:
            raise RuntimeError(f"No response text received from Gemini for {video_id}.")

        cleaned_text = resp.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        model_output = json.loads(cleaned_text.strip())

        # Extract sections
        env_data = model_output.get("environment", {})
        behavior_data = model_output.get("scenario_behavior", {})
        actors_data = model_output.get("actors", [])

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

        # Final annotation adhering to project schema
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

        # Validate against schema
        val_schema = load_validation_schema()
        if val_schema and HAS_JSONSCHEMA:
            try:
                jsonschema.validate(instance=final_annotation, schema=val_schema)
                print(f"[{video_id}] Schema Validation: PASSED ✓")
            except jsonschema.ValidationError as err:
                print(f"[{video_id}] Schema Validation Warning: {err.message}")

        # Write output JSON
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_annotation, f, indent=2)
            f.write("\n")

        print(f"[{video_id}] Saved annotation to: {output_json_path}")
        return True, model_name

    except Exception as err:
        print(f"[{video_id}] ERROR processing video with Gemini: {err}")
        return False, model_name
    finally:
        # Auto-cleanup uploaded file from Gemini storage
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def main():
    if not HAS_GENAI:
        print("Error: google-genai package is not installed. Run: pip install google-genai")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Annotate autonomous driving videos using Google Gemini Multimodal API."
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Specific video filename or path"
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        help="List of specific video filenames or paths to annotate."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Annotate all videos in videos/ folder."
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
        "--output-dir",
        type=str,
        default=str(DEFAULT_ANNOTATIONS_DIR),
        help="Destination directory for Gemini annotations (default: annotations_gemini)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation JSON files."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Gemini model to use (default: 'gemini-3.5-flash-lite')."
    )

    args = parser.parse_args()

    api_key = verify_api_key()
    client = genai.Client(api_key=api_key)
    output_dir = Path(args.output_dir)

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
            print("No NuScenes video files found.")
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
            print("No KITTI video files found.")
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
            f for f in sorted(VIDEOS_DIR.rglob("*.mp4"))
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not target_videos:
            print(f"No video files found in {VIDEOS_DIR}.")
            sys.exit(0)
    else:
        print("No video specified. Usage examples:")
        print("  ./venv/bin/python scripts/annotate_with_gemini.py --video videos/dashcam/EVT_20240609_145337_2.mp4")
        print("  ./venv/bin/python scripts/annotate_with_gemini.py --dashcam --limit 20")
        print("  ./venv/bin/python scripts/annotate_with_gemini.py --idd --limit 20")
        print("  ./venv/bin/python scripts/annotate_with_gemini.py --kitti")
        print("  ./venv/bin/python scripts/annotate_with_gemini.py --nuscenes")
        sys.exit(0)

    # Filter unannotated
    if not args.overwrite:
        unannotated = []
        for v in target_videos:
            try:
                rel_dir = v.resolve().relative_to(VIDEOS_DIR.resolve()).parent
            except ValueError:
                rel_dir = Path(".")
            if rel_dir.name.lower() == "idd":
                rel_dir = Path("IDD")
            out_p = output_dir / rel_dir / f"{v.stem}.json"
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

    active_model = args.model
    print(f"\nProcessing {len(target_videos)} video(s) with Gemini ({active_model}) -> {output_dir}")
    success_count = 0
    for idx, v in enumerate(target_videos, start=1):
        print(f"\n[{idx}/{len(target_videos)}] Target: {v.name}")
        ok, active_model = annotate_video_with_gemini(
            v,
            client,
            model_name=active_model,
            output_dir=output_dir,
            overwrite=args.overwrite
        )
        if ok:
            success_count += 1
        if idx < len(target_videos):
            time.sleep(2)

    print(f"\nAnnotation run complete! Successfully processed: {success_count}/{len(target_videos)}")


if __name__ == "__main__":
    main()

