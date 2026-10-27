#!/usr/bin/env python3
"""
annotate_with_qwen.py

Automated autonomous driving video scenario and behavior annotation pipeline
powered by local/remote Qwen-VL (qwen2.5vl:7b) via Ollama inference server.

Strictly adheres to schema/annotation_schema.json and generates outputs in:
  annotations_qwen/<dataset_subfolder>/<video_id>.json
"""

import os
import sys
import time
import json
import base64
import tempfile
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
DEFAULT_ANNOTATIONS_DIR = BASE_DIR / "annotations_qwen"
SCHEMA_PATH = BASE_DIR / "schema" / "annotation_schema.json"

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11435")
DEFAULT_MODEL = "qwen2.5vl:7b"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

TAXONOMY_PROMPT = """
You are an expert autonomous driving scenario annotator.
Analyze this sequence of temporal video frames from an ego vehicle perspective and annotate the primary event, environment, and all interacting actors according to this strict taxonomy:

1. EVENT TYPE (select exactly one):
- 'Traffic Control': Traffic light changes, traffic sign recognition (speed limits, stop signs, yield, one-way, no entry).
- 'Vehicle Interaction': Cut-ins, cut-outs, lane changes, sudden stops from leading vehicles, vehicle overtaking, following, merging.
- 'Road & Environment': Changes in road conditions (potholes, speed bumps, wet road, debris), weather shifts affecting visibility.
- 'Obstacle Detection': Moving obstacles or stationary objects (parked cars, cones, fallen trees, barriers, toll/security gates).
- 'Animal & Pedestrian': Animals crossing/on the road, pedestrians crossing, on sidewalks, or walking along roadways.
- 'Sudden & Unexpected': Emergency vehicles (ambulance, fire truck), crashes, breakdowns, erratic driving.

2. EVENT PATTERN (select exactly one):
- 'Co-linear': Events on the same route/lane as ego vehicle (in line with ego vehicle travel).
- 'Cross-linear': Events crossing the ego vehicle route perpendicularly or from sides/intersections.
- 'Causal': Direct trigger-action cause-effect (e.g. lead car brakes -> ego brakes; gate closed -> ego stops).
- 'Likelihood': Anticipating potential future hazard (e.g. pedestrian standing at curb likely to cross).

3. SEVERITY (select exactly one):
- 'Low': Normal driving, distant objects, minimal or routine speed adjustment.
- 'Medium': Noticeable speed adjustment, following in dense traffic, moderate caution.
- 'Moderate': Heavy braking, stopped behind traffic/gate, sharp maneuvering.
- 'High': Emergency stop, near-miss, sudden collision avoidance maneuver.

4. ENVIRONMENT:
- road_type: 'Highway' | 'Urban' | 'Rural' | 'Semi-urban' | 'Unknown'
- intersection: true | false
- traffic_control: 'Traffic signal' | 'Stop sign' | 'Yield' | 'None' | 'Unknown'
- weather: 'Clear' | 'Rain' | 'Fog' | 'Cloudy'
- lighting: 'Day' | 'Night' | 'Dawn/Dusk'

5. EGO MANEUVER (select exactly one):
'Go straight' | 'Follow' | 'Stop' | 'Overtake' | 'Left Turn' | 'Right Turn' | 'Lane change' | 'Cut-in' | 'Cut-out' | 'Emergency brake' | 'Yield' | 'Other'
(Note: Use 'Follow' if keeping pace behind a lead vehicle. Use 'Stop' if vehicle halts or waits at a light/gate/congestion. Use 'Overtake' if passing a slower vehicle.)

6. ACTORS (IMPORTANT: List ALL distinct road users and objects visible across the scene):
Scan all lanes (front, left, right, oncoming, roadside). For EACH distinct actor group observed:
- class: string ('car', 'truck', 'motorcycle', 'cyclist', 'pedestrian', 'bus', 'animal', 'barrier', 'gate')
- count: integer (>= 1)
- relative_position: string ('front', 'front_left', 'front_right', 'left', 'right', 'rear', 'crossing')
Do NOT omit secondary vehicles like trucks in adjacent lanes, motorcycles, or pedestrians!

7. KEY EVENT:
Write a descriptive, unique 1-sentence summary explaining specifically what happens in THIS clip (e.g. mention pedestrians, cyclists, motorcycles, trucks, turns, lead vehicles, roadside objects). Do NOT copy boilerplate text.

8. TIME WINDOW:
- start_sec: string in "MM:SS" format (e.g. "00:00")
- end_sec: string in "MM:SS" format (e.g. "00:10")

Return ONLY a valid JSON object matching this structure:
{
  "environment": {
    "road_type": "Urban",
    "intersection": false,
    "traffic_control": "None",
    "weather": "Clear",
    "lighting": "Day"
  },
  "scenario_behavior": {
    "event_type": "Vehicle Interaction",
    "event_pattern": "Co-linear",
    "severity": "Low",
    "ego_maneuver": "Follow",
    "key_event": "Unique natural language sentence explaining what specifically happens in this video clip",
    "time_window": {
      "start_sec": "00:00",
      "end_sec": "00:10"
    }
  },
  "actors": [
    {
      "class": "car",
      "count": 1,
      "relative_position": "front"
    }
  ]
}
"""

VALID_ROAD_TYPES = ["Urban", "Semi-urban", "Rural", "Highway", "Unknown"]
VALID_TRAFFIC_CONTROLS = ["Traffic signal", "Stop sign", "Yield", "None", "Unknown"]
VALID_WEATHERS = ["Clear", "Rain", "Fog", "Cloudy"]
VALID_LIGHTINGS = ["Day", "Night", "Dawn/Dusk"]
VALID_EVENT_TYPES = [
    "Traffic Control", "Vehicle Interaction", "Road & Environment",
    "Obstacle Detection", "Animal & Pedestrian", "Sudden & Unexpected"
]
VALID_EVENT_PATTERNS = ["Co-linear", "Cross-linear", "Causal", "Likelihood"]
VALID_SEVERITIES = ["Low", "Medium", "Moderate", "High"]
VALID_MANEUVERS = [
    "Go straight", "Left Turn", "Right Turn", "Lane change", "Cut-in",
    "Cut-out", "Overtake", "Emergency brake", "Yield", "Follow", "Stop", "Other"
]


def load_validation_schema():
    if SCHEMA_PATH.exists():
        try:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read schema file: {e}")
    return None


def normalize_choice(value, valid_choices, default):
    if not value or not isinstance(value, str):
        return default
    v_clean = value.strip().lower()

    # Specific semantic mappings to schema enums
    if "overcast" in v_clean or "cloud" in v_clean:
        if "Cloudy" in valid_choices:
            return "Cloudy"
    if "traffic light" in v_clean or "signal" in v_clean:
        if "Traffic signal" in valid_choices:
            return "Traffic signal"
    if "dawn" in v_clean or "dusk" in v_clean or "twilight" in v_clean:
        if "Dawn/Dusk" in valid_choices:
            return "Dawn/Dusk"
    if "rain" in v_clean or "wet" in v_clean or "storm" in v_clean or "snow" in v_clean:
        if "Rain" in valid_choices:
            return "Rain"
    if "fog" in v_clean or "mist" in v_clean:
        if "Fog" in valid_choices:
            return "Fog"

    for choice in valid_choices:
        if v_clean == choice.lower():
            return choice
    for choice in valid_choices:
        if choice.lower() in v_clean or v_clean in choice.lower():
            return choice
    return default


def extract_video_frames(video_path: Path, max_frames: int = 8, scale_width: int = 512, q_val: int = 4) -> list[str]:
    """
    Extracts evenly spaced keyframes from video and returns a list of base64 JPEG strings.
    Optimized for fast GPU vision encoding without overflowing Ollama context limits.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"fps=1,scale={scale_width}:-1",
            "-q:v", str(q_val),
            f"{tmpdir}/frame_%04d.jpg"
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg failed to extract frames from {video_path.name}")

        frame_files = sorted(list(Path(tmpdir).glob("frame_*.jpg")))
        if not frame_files:
            raise RuntimeError(f"No frames extracted from {video_path.name}")

        # Subsample up to max_frames evenly spaced
        if len(frame_files) > max_frames:
            step = len(frame_files) / float(max_frames)
            selected_indices = [int(i * step) for i in range(max_frames)]
            frame_files = [frame_files[idx] for idx in selected_indices]

        b64_frames = []
        for f in frame_files:
            with open(f, "rb") as img_f:
                b64_frames.append(base64.b64encode(img_f.read()).decode("utf-8"))

        return b64_frames


def query_ollama(api_url: str, model_name: str, prompt: str, images: list[str], timeout: int = 120) -> dict:
    """
    Sends multimodal request with base64 images to Ollama /api/generate endpoint.
    Explicitly bounds num_ctx to 12288 so model executes 100% in GPU VRAM without CPU swapping.
    """
    url = f"{api_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": images,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 12288
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def annotate_video_with_qwen(
    video_path: Path,
    api_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL,
    output_dir: Path = DEFAULT_ANNOTATIONS_DIR,
    overwrite: bool = False
) -> bool:
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
        return True

    # Detect dataset
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

    try:
        print(f"\n[{video_id}] Extracting temporal frames with FFmpeg...")
        t0 = time.time()
        images = extract_video_frames(video_path, max_frames=8, scale_width=512)
        print(f"[{video_id}] Extracted {len(images)} frames ({time.time() - t0:.1f}s)")

        print(f"[{video_id}] Running multimodal scenario understanding with {model_name} on {api_url}...")
        t_infer = time.time()
        
        # Retry logic for network/transient hiccups
        max_retries = 3
        res = None
        for attempt in range(max_retries):
            try:
                res = query_ollama(api_url, model_name, TAXONOMY_PROMPT, images, timeout=120)
                if res and res.get("response"):
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{video_id}] Ollama request glitch: {e}. Retrying in 5s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(5)
                else:
                    raise e

        if not res or not res.get("response"):
            raise RuntimeError(f"Empty response from Ollama for {video_id}")

        print(f"[{video_id}] Inference complete ({time.time() - t_infer:.1f}s)")
        raw_response = res.get("response", "").strip()

        # Parse JSON
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:]
        if raw_response.endswith("```"):
            raw_response = raw_response[:-3]
        model_output = json.loads(raw_response.strip())

        # Extract sections
        env_data = model_output.get("environment", {})
        behavior_data = model_output.get("scenario_behavior", {})
        actors_data = model_output.get("actors", [])

        # Normalize time_window
        tw = behavior_data.get("time_window", {})
        if not isinstance(tw, dict):
            tw = {"start_sec": "00:00", "end_sec": "00:10"}
        if "start_sec" not in tw:
            tw["start_sec"] = "00:00"
        if "end_sec" not in tw:
            tw["end_sec"] = "00:10"
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
                "key_event": str(behavior_data.get("key_event", f"Driving scenario in {video_id}")).strip() or f"Driving scenario in {video_id}",
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
        return True

    except Exception as err:
        print(f"[{video_id}] ERROR processing video with Qwen: {err}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Annotate autonomous driving videos using Qwen2.5-VL via Ollama inference server."
    )
    parser.add_argument("--video", "-v", type=str, help="Specific video filename or path")
    parser.add_argument("--videos", nargs="+", type=str, help="List of specific video filenames or paths to annotate.")
    parser.add_argument("--all", action="store_true", help="Annotate all videos in videos/ folder.")
    parser.add_argument("--nuscenes", action="store_true", help="Annotate all NuScenes videos.")
    parser.add_argument("--kitti", action="store_true", help="Annotate all KITTI videos.")
    parser.add_argument("--dashcam", action="store_true", help="Annotate all Dashcam videos.")
    parser.add_argument("--idd", action="store_true", help="Annotate all IDD videos.")
    parser.add_argument("--limit", type=int, default=None, help="Limit maximum number of videos to annotate in this run.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_ANNOTATIONS_DIR),
        help="Destination directory for Qwen annotations (default: annotations_qwen)."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotation JSON files.")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: '{DEFAULT_OLLAMA_URL}')."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama vision model to use (default: '{DEFAULT_MODEL}')."
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    target_videos = []
    if args.video:
        target_videos.append(Path(args.video).resolve())
    elif args.videos:
        target_videos.extend([Path(v).resolve() for v in args.videos])
    elif args.nuscenes:
        target_videos = [
            f for f in sorted(VIDEOS_DIR.rglob("*.mp4"))
            if f.is_file() and ("nuscenes" in str(f).lower() or f.name.startswith("scene-"))
        ]
        if not target_videos:
            print("No NuScenes video files found in videos/nuscenes dataset/ or starting with 'scene-'.")
            sys.exit(0)
    elif args.kitti:
        target_videos = [
            f for f in sorted(VIDEOS_DIR.rglob("*.mp4"))
            if f.is_file() and ("kitti" in str(f).lower() or f.name.startswith("2011_"))
        ]
        if not target_videos:
            print("No KITTI video files found in videos/kitti dataset/ or starting with '2011_'.")
            sys.exit(0)
    elif args.dashcam:
        target_videos = [
            f for f in sorted(VIDEOS_DIR.rglob("*.mp4"))
            if f.is_file() and "dashcam" in str(f).lower()
        ]
        if not target_videos:
            print("No Dashcam video files found in videos/dashcam/.")
            sys.exit(0)
    elif args.idd:
        candidates = list(VIDEOS_DIR.rglob("*.mp4"))
        target_videos = []
        seen = set()
        for f in sorted(candidates):
            rf = f.resolve()
            if rf not in seen and f.is_file():
                if "idd" in str(rf).lower() or f.name.startswith("idd_") or "combined" in f.name.lower():
                    seen.add(rf)
                    target_videos.append(f)
        if not target_videos:
            candidates = list((VIDEOS_DIR / "IDD").glob("*.mp4")) + list((VIDEOS_DIR / "idd").glob("*.mp4"))
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
        print("  ./venv/bin/python scripts/annotate_with_qwen.py --video videos/dashcam/EVT_20240609_145337_2.mp4")
        print("  ./venv/bin/python scripts/annotate_with_qwen.py --dashcam --limit 20")
        print("  ./venv/bin/python scripts/annotate_with_qwen.py --idd --limit 20")
        print("  ./venv/bin/python scripts/annotate_with_qwen.py --kitti")
        print("  ./venv/bin/python scripts/annotate_with_qwen.py --nuscenes")
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

    print(f"\nProcessing {len(target_videos)} video(s) with Qwen ({args.model}) -> {output_dir}")
    success_count = 0
    for idx, v in enumerate(target_videos, start=1):
        print(f"\n[{idx}/{len(target_videos)}] Target: {v.name}")
        ok = annotate_video_with_qwen(
            v,
            api_url=args.url,
            model_name=args.model,
            output_dir=output_dir,
            overwrite=args.overwrite
        )
        if ok:
            success_count += 1

    print(f"\nAnnotation run complete! Successfully processed: {success_count}/{len(target_videos)}")


if __name__ == "__main__":
    main()

