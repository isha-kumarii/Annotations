#!/usr/bin/env python3
"""
annotate_v2_gemini.py

Automated autonomous driving video scenario and behavior annotation pipeline (V2)
powered by Google Gemini Multimodal API (gemini-3.5-flash-lite).

Strictly adheres to schema/annotation_schema_v2.json and outputs to:
  annotations_v2/<dataset_subfolder>/<video_id>.json
"""

import os
import sys
import time
import json
import queue
import argparse
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
DEFAULT_ANNOTATIONS_DIR = BASE_DIR / "annotations_v2"
SCHEMA_PATH = BASE_DIR / "schema" / "annotation_schema_v2.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

TAXONOMY_PROMPT_V2 = """
You are an expert autonomous driving scenario annotator.
Analyze this video recording from an ego vehicle perspective and annotate the primary event, road network, environmental conditions, ego vehicle behavior, and interacting NPC actors according to this exact taxonomy:

1. ROAD NETWORK:
- road_type: 'highway', 'rural', 'urban', or 'semi-urban'
- road_layout: 'straight road', 'curved road', 'intersection', or 'roundabout'
- no_of_lanes: integer number of lanes (0, 1, 2, 3, 4, ...)
- intersection_control: 'signalized', 'unsignalized', or 'none'
- traffic_sign: list of observed traffic signs, e.g. ["stop", "no entry", "merge", "yield", "speed limit"] (empty list [] if none)
- traffic_light: active signal facing ego vehicle ('red', 'green', 'yellow', or 'none')

2. ENVIRONMENTAL CONDITIONS:
- weather: 'clear', 'overcast', 'rain', 'fog', 'snow', 'wet', or 'haze'
- visibility: 'Good', 'Moderate', 'Poor', or 'Unknown'
- lighting: 'Daylight', 'Twilight', 'Night', or 'Unknown'

3. EGO VEHICLE:
- maneuver (select exactly one):
  'go straight', 'follow', 'turn left', 'turn right', 'lane change left', 'lane change right',
  'stop', 'yield', 'U-turn', 'overtake', 'emergency brake', 'cut-in', 'cut-out'

4. NPC ACTORS:
List all interacting or notable road actors (vehicles, vulnerable road users, animals) with:
- actor_type: 'pedestrian', 'animal', 'car', 'bicycle', 'truck', 'bus', 'motorbike', 'auto_rickshaw'
- maneuver: 'go straight', 'follow', 'turn left', 'turn right', 'lane change left', 'lane change right',
  'stop', 'yield', 'crossing road left to right', 'crossing road right to left', 'U-turn', 'overtake', 'stationary'

5. EVENT-DETAILS:
- event_description: Clear, unique natural language description summarizing the key driving event.
- event_class (select exactly one):
  * 'Traffic Control': Incidents involving detection of traffic lights and signs regulating traffic flow.
  * 'Vehicle Interaction': Two or more vehicles directly influencing each other's movement.
  * 'Road and Environment': Changes in road/environmental conditions affecting navigation (potholes, wet surface, weather).
  * 'Obstacle Detection': Static or moving objects impeding navigation (parked cars, cones, barriers, gates).
  * 'Animal and Pedestrian': Pedestrian or animal movements on road necessitating slowing down or stopping.
  * 'Sudden and Unexpected': Unpredicted events necessitating quick response (emergency vehicles, erratic driving, near-misses).
- event_pattern (select exactly one):
  * 'Co-linear': Events in the direction of movement of the ego vehicle (same lane/travel path).
  * 'Cross-linear': Events not directly in line with movement but potentially impacting travel (crossing traffic, sides, intersections).
  * 'Causal': Direct trigger-action cause-effect (e.g. lead car brakes -> ego brakes).
  * 'Likelihood': Predicting likelihood of future events based on current cues (e.g. pedestrian standing at curb likely to cross).
- event_severity (select exactly one):
  * 'High': Critical issues requiring immediate attention or collision avoidance.
  * 'medium': Issues needing attention but not immediate emergency response.
  * 'low': Minor, routine non-critical driving events.

Return ONLY a valid JSON object matching this exact structure:
{
  "road_network": {
    "road_type": "urban",
    "road_layout": "intersection",
    "no_of_lanes": 2,
    "intersection_control": "signalized",
    "traffic_sign": ["stop"],
    "traffic_light": "green"
  },
  "environmental_conditions": {
    "weather": "clear",
    "visibility": "Good",
    "lighting": "Daylight"
  },
  "ego_vehicle": {
    "maneuver": "go straight"
  },
  "npc_actors": [
    {
      "actor_type": "car",
      "maneuver": "follow"
    }
  ],
  "event_details": {
    "event_description": "...",
    "event_class": "Vehicle Interaction",
    "event_pattern": "Co-linear",
    "event_severity": "low"
  }
}
"""

VALID_ROAD_TYPES = ["highway", "rural", "urban", "semi-urban"]
VALID_ROAD_LAYOUTS = ["straight road", "curved road", "intersection", "roundabout"]
VALID_INTERSECTION_CONTROLS = ["signalized", "unsignalized", "none"]
VALID_TRAFFIC_LIGHTS = ["red", "green", "yellow", "none"]
VALID_WEATHERS = ["clear", "overcast", "rain", "fog", "snow", "wet", "haze"]
VALID_VISIBILITIES = ["Good", "Moderate", "Poor", "Unknown"]
VALID_LIGHTINGS = ["Daylight", "Twilight", "Night", "Unknown"]
VALID_EGO_MANEUVERS = [
    "go straight", "follow", "turn left", "turn right", "lane change left",
    "lane change right", "stop", "yield", "U-turn", "overtake",
    "emergency brake", "cut-in", "cut-out"
]
VALID_NPC_MANEUVERS = [
    "go straight", "follow", "turn left", "turn right", "lane change left",
    "lane change right", "stop", "yield", "crossing road left to right",
    "crossing road right to left", "U-turn", "overtake", "stationary", "unknown"
]
VALID_EVENT_CLASSES = [
    "Traffic Control", "Vehicle Interaction", "Road and Environment",
    "Obstacle Detection", "Animal and Pedestrian", "Sudden and Unexpected"
]
VALID_EVENT_PATTERNS = ["Co-linear", "Cross-linear", "Causal", "Likelihood", "Predictive"]
VALID_SEVERITIES = ["High", "medium", "low"]


def get_api_keys() -> list:
    keys = []
    # Check numbered keys first (GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.)
    for i in range(1, 10):
        val = os.getenv(f"GEMINI_API_KEY_{i}")
        if val and val.strip() and val.strip() not in keys:
            keys.append(val.strip())
    # Also check unnumbered key
    default_val = os.getenv("GEMINI_API_KEY")
    if default_val and default_val.strip() and default_val.strip() not in keys:
        keys.append(default_val.strip())
    return keys


def verify_api_key() -> str:
    keys = get_api_keys()
    if not keys:
        print("Error: No GEMINI_API_KEY found in environment or .env file.")
        sys.exit(1)
    return keys[0]


def load_validation_schema():
    if not HAS_JSONSCHEMA or not SCHEMA_PATH.exists():
        return None
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load schema: {e}")
        return None


def normalize_choice(val, valid_choices, default):
    if val is None:
        return default
    v_clean = str(val).strip().lower()
    for choice in valid_choices:
        if v_clean == choice.lower():
            return choice
    for choice in valid_choices:
        if choice.lower() in v_clean or v_clean in choice.lower():
            return choice
    return default


def clean_gemini_json_response(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Empty response text from Gemini API")
    text = raw_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        last_backticks = text.rfind("```")
        if first_newline != -1 and last_backticks != -1:
            text = text[first_newline + 1:last_backticks].strip()
    return json.loads(text)


def detect_dataset_name(video_path: Path) -> str:
    name_str = str(video_path).lower()
    stem = video_path.stem
    if "dashcam" in name_str or stem.startswith("EVT_") or stem.startswith("REC_"):
        return "Dashcam"
    elif stem.startswith("scene-") or "nuscenes" in name_str:
        return "NuScenes"
    elif stem.startswith("2011_") or "kitti" in name_str:
        return "KITTI"
    elif "idd" in name_str or "combined" in stem.lower():
        return "IDD"
    return "Driving"


def annotate_video_v2(
    video_path: Path,
    client: genai.Client,
    worker_id: int = 1,
    model_name: str = "gemini-3.5-flash-lite",
    output_dir: Path = DEFAULT_ANNOTATIONS_DIR,
    overwrite: bool = False,
    delete_source_after: bool = False
) -> bool:
    video_id = video_path.stem
    tag = f"[Worker {worker_id} | {video_id}]"
    dataset_name = detect_dataset_name(video_path)

    if dataset_name == "NuScenes":
        rel_dir = Path("nuscenes dataset")
    elif dataset_name == "IDD":
        rel_dir = Path("IDD")
    elif dataset_name == "KITTI":
        rel_dir = Path("kitti dataset")
    elif dataset_name == "Dashcam":
        rel_dir = Path("dashcam")
    else:
        rel_dir = Path(dataset_name.lower())

    output_json_path = output_dir / rel_dir / f"{video_id}.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    if output_json_path.exists() and not overwrite:
        print(f"Skipping {video_id}: Annotation already exists ({output_json_path}).")
        return True

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

        print(f"[{video_id}] Running scenario understanding with {model_name}...")
        resp = None
        for attempt in range(4):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, TAXONOMY_PROMPT_V2],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                if resp and resp.text:
                    break
            except Exception as e:
                err_s = str(e).lower()
                if "429" in err_s or "resource_exhausted" in err_s:
                    if "daily" in err_s or "day" in err_s or "generaterequestsperday" in err_s:
                        print(f"[{video_id}] Daily quota limit reached: {e}")
                        return False
                    wait_sec = (attempt + 1) * 10
                    print(f"[{video_id}] Transient rate limit hit, pausing {wait_sec}s (retry {attempt+1}/3)...")
                    time.sleep(wait_sec)
                else:
                    if attempt == 3:
                        raise e
                    time.sleep(5)

        if not resp or not resp.text:
            raise RuntimeError(f"Empty response from Gemini API for {video_id}")

        data = clean_gemini_json_response(resp.text)

        # Parse & normalize fields
        rn = data.get("road_network", {})
        env = data.get("environmental_conditions", {})
        ego = data.get("ego_vehicle", {})
        actors = data.get("npc_actors", [])
        evt = data.get("event_details", {})

        # Road network lanes
        lanes = rn.get("no_of_lanes", 2)
        try:
            lanes = int(lanes)
        except (ValueError, TypeError):
            lanes = 2

        # Traffic signs
        signs = rn.get("traffic_sign", [])
        if not isinstance(signs, list):
            signs = [str(signs)] if signs else []

        # NPC Actors normalization
        cleaned_actors = []
        if isinstance(actors, list):
            for a in actors:
                if isinstance(a, dict) and "actor_type" in a:
                    cleaned_actors.append({
                        "actor_type": str(a.get("actor_type", "car")).lower().strip(),
                        "maneuver": normalize_choice(a.get("maneuver"), VALID_NPC_MANEUVERS, "follow")
                    })
        if not cleaned_actors:
            cleaned_actors = [{"actor_type": "car", "maneuver": "follow"}]

        # Construct final V2 JSON
        final_annotation = {
            "metadata": {
                "video_id": video_id,
                "dataset": dataset_name,
                "video_path": str(video_path)
            },
            "road_network": {
                "road_type": normalize_choice(rn.get("road_type"), VALID_ROAD_TYPES, "urban"),
                "road_layout": normalize_choice(rn.get("road_layout"), VALID_ROAD_LAYOUTS, "straight road"),
                "no_of_lanes": lanes,
                "intersection_control": normalize_choice(rn.get("intersection_control"), VALID_INTERSECTION_CONTROLS, "none"),
                "traffic_sign": signs,
                "traffic_light": normalize_choice(rn.get("traffic_light"), VALID_TRAFFIC_LIGHTS, "none")
            },
            "environmental_conditions": {
                "weather": normalize_choice(env.get("weather"), VALID_WEATHERS, "clear"),
                "visibility": normalize_choice(env.get("visibility"), VALID_VISIBILITIES, "Good"),
                "lighting": normalize_choice(env.get("lighting"), VALID_LIGHTINGS, "Daylight")
            },
            "ego_vehicle": {
                "maneuver": normalize_choice(ego.get("maneuver"), VALID_EGO_MANEUVERS, "go straight")
            },
            "npc_actors": cleaned_actors,
            "event_details": {
                "event_description": str(evt.get("event_description", f"Driving scenario in {video_id}")).strip(),
                "event_class": normalize_choice(evt.get("event_class"), VALID_EVENT_CLASSES, "Vehicle Interaction"),
                "event_pattern": "Likelihood" if str(evt.get("event_pattern", "")).strip().lower() in ("predictive", "likelihood") else normalize_choice(evt.get("event_pattern"), VALID_EVENT_PATTERNS, "Co-linear"),
                "event_severity": normalize_choice(evt.get("event_severity"), VALID_SEVERITIES, "low")
            }
        }

        # Schema Validation
        val_schema = load_validation_schema()
        if val_schema and HAS_JSONSCHEMA:
            try:
                jsonschema.validate(instance=final_annotation, schema=val_schema)
                print(f"[{video_id}] Schema Validation: PASSED ✓")
            except jsonschema.ValidationError as err:
                print(f"[{video_id}] Schema Validation Warning: {err.message}")

        # Save output JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_annotation, f, indent=2)
            f.write("\n")
        print(f"[{video_id}] Saved annotation to: {output_json_path}")

        # Optional auto-delete video to maintain disk headroom
        if delete_source_after:
            try:
                video_path.unlink()
                print(f"[{video_id}] Deleted source video {video_path.name} to free disk space.")
            except Exception as e:
                print(f"[{video_id}] Warning: Could not delete {video_path.name}: {e}")

        return True

    except Exception as err:
        print(f"[{video_id}] ERROR processing video: {err}")
        return False
    finally:
        if uploaded_file and hasattr(uploaded_file, "name"):
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def main():
    if not HAS_GENAI:
        print("Error: google-genai is not installed. Run: pip install google-genai")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Annotate autonomous driving videos using V2 schema with Gemini Flash-Lite.")
    parser.add_argument("--dir", type=str, default=str(VIDEOS_DIR), help="Directory containing videos to annotate.")
    parser.add_argument("--video", type=str, help="Single video file path to annotate.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of videos to annotate in this batch.")
    parser.add_argument("--workers", type=int, default=0, help="Number of concurrent workers (default: auto, matches number of API keys).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotations.")
    parser.add_argument("--delete-source", action="store_true", help="Delete video file after successful annotation to save disk space.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_ANNOTATIONS_DIR), help="Output annotations directory.")
    args = parser.parse_args()

    api_keys = get_api_keys()
    if not api_keys:
        print("Error: No GEMINI_API_KEY found in environment or .env file.")
        sys.exit(1)

    print(f"Loaded {len(api_keys)} Gemini API key(s) from environment/.env.")

    # Client pool queue for thread-safe assignment
    client_pool = queue.Queue()
    for idx, k in enumerate(api_keys, start=1):
        try:
            c = genai.Client(api_key=k)
            client_pool.put((idx, c))
        except Exception as e:
            print(f"Warning: Could not initialize client with key #{idx}: {e}")

    num_workers = args.workers if args.workers > 0 else client_pool.qsize()
    num_workers = max(1, min(num_workers, client_pool.qsize()))
    print(f"Running with {num_workers} concurrent worker(s) across {client_pool.qsize()} available key(s).\n")

    output_dir = Path(args.output_dir)

    target_videos = []
    if args.video:
        vp = Path(args.video)
        if vp.is_file():
            target_videos.append(vp)
        else:
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
    else:
        search_dir = Path(args.dir)
        candidates = sorted(list(search_dir.rglob("*.mp4")))
        seen = set()
        for f in candidates:
            rf = f.resolve()
            if rf not in seen and f.is_file():
                seen.add(rf)
                target_videos.append(f)

    # Filter unannotated
    if not args.overwrite:
        unannotated = []
        for v in target_videos:
            ds = detect_dataset_name(v)
            if ds == "NuScenes":
                rel = Path("nuscenes dataset")
            elif ds == "KITTI":
                rel = Path("kitti dataset")
            elif ds == "IDD":
                rel = Path("IDD")
            elif ds == "Dashcam":
                rel = Path("dashcam")
            else:
                rel = Path(ds.lower())
            out_p = output_dir / rel / f"{v.stem}.json"
            if not out_p.exists():
                unannotated.append(v)
        print(f"Total found: {len(target_videos)} | Unannotated: {len(unannotated)} | Already done: {len(target_videos) - len(unannotated)}")
        target_videos = unannotated

    if args.limit > 0:
        target_videos = target_videos[:args.limit]
        print(f"Limiting to first {len(target_videos)} videos for this batch.")

    if not target_videos:
        print("No unannotated videos to process.")
        sys.exit(0)

    print(f"\nProcessing {len(target_videos)} video(s) -> {output_dir}")

    def process_one(item):
        idx, v = item
        worker_id, client = client_pool.get()
        try:
            print(f"\n[{idx}/{len(target_videos)}] [Worker {worker_id}] Starting {v.name}...")
            ok = annotate_video_v2(
                v,
                client,
                worker_id=worker_id,
                output_dir=output_dir,
                overwrite=args.overwrite,
                delete_source_after=args.delete_source
            )
            return ok
        finally:
            client_pool.put((worker_id, client))

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        items = list(enumerate(target_videos, start=1))
        results = executor.map(process_one, items)
        for ok in results:
            if ok:
                success_count += 1

    print(f"\nBatch complete! Successfully annotated: {success_count}/{len(target_videos)}")


if __name__ == "__main__":
    main()

