#!/usr/bin/env python3
"""
process_nuscenes_dataset.py

Parses and compiles all 1,000 NuScenes videos from 'videos/Nuscenes Dataset'
using the Gemini event breakdowns, scene captions, and object timestamps
into strict V2 JSON annotations (annotations_v2/nuscenes dataset/<video_id>.json).
"""

import re
import json
from pathlib import Path
import jsonschema

BASE_DIR = Path(__file__).resolve().parent.parent
NUSCENES_DIR = BASE_DIR / "videos" / "Nuscenes Dataset"
OUTPUT_DIR = BASE_DIR / "annotations_v2" / "nuscenes dataset"
SCHEMA_PATH = BASE_DIR / "schema" / "annotation_schema_v2.json"

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
VALID_EVENT_PATTERNS = ["Co-linear", "Cross-linear", "Causal", "Predictive"]
VALID_SEVERITIES = ["High", "medium", "low"]


def extract_lanes(caption: str) -> int:
    cap = caption.lower()
    if "one-lane" in cap or "single-lane" in cap or "1-lane" in cap:
        return 1
    if "two-lane" in cap or "2-lane" in cap:
        return 2
    if "three-lane" in cap or "3-lane" in cap:
        return 3
    if "four-lane" in cap or "4-lane" in cap:
        return 4
    if "multi-lane" in cap:
        return 3
    return 2


def extract_road_layout(caption: str, events: list) -> str:
    combined = (caption + " " + " ".join([e.get("event_description", "") for e in events])).lower()
    if "roundabout" in combined:
        return "roundabout"
    if "intersection" in combined or "junction" in combined or "crossroad" in combined:
        return "intersection"
    if "curve" in combined or "curved" in combined or "bend" in combined or "turn" in combined:
        return "curved road"
    return "straight road"


def extract_road_type(caption: str) -> str:
    cap = caption.lower()
    if "highway" in cap or "expressway" in cap or "freeway" in cap:
        return "highway"
    if "rural" in cap or "countryside" in cap:
        return "rural"
    if "industrial" in cap or "suburban" in cap or "residential" in cap:
        return "semi-urban"
    return "urban"


def extract_traffic_light(caption: str, events: list) -> tuple[str, str]:
    """Returns (intersection_control, traffic_light)"""
    combined = (caption + " " + " ".join([e.get("event_description", "") for e in events])).lower()
    if "red traffic light" in combined or "red light" in combined:
        return "signalized", "red"
    if "yellow traffic light" in combined or "amber" in combined:
        return "signalized", "yellow"
    if "green traffic light" in combined or "green light" in combined:
        return "signalized", "green"
    if "traffic light" in combined or "traffic signal" in combined or "signalized" in combined:
        return "signalized", "green"
    if "intersection" in combined:
        return "unsignalized", "none"
    return "none", "none"


def extract_traffic_signs(caption: str) -> list[str]:
    cap = caption.lower()
    signs = []
    if "stop sign" in cap or "stops at a stop" in cap:
        signs.append("stop")
    if "yield sign" in cap or "yield" in cap:
        signs.append("yield")
    if "no entry" in cap:
        signs.append("no entry")
    if "traffic cone" in cap or "cones" in cap:
        signs.append("traffic cones")
    if "directional sign" in cap or "guide sign" in cap:
        signs.append("directional signs")
    if "construction" in cap:
        signs.append("construction signs")
    return signs


def extract_weather(caption: str) -> str:
    cap = caption.lower()
    if "rain" in cap or "raining" in cap or "wet" in cap:
        return "rain"
    if "fog" in cap or "mist" in cap:
        return "fog"
    if "snow" in cap or "snowing" in cap:
        return "snow"
    if "haze" in cap or "smog" in cap:
        return "haze"
    if "overcast" in cap or "cloudy" in cap:
        return "overcast"
    return "clear"


def extract_lighting(caption: str) -> str:
    cap = caption.lower()
    if "night" in cap or "dark" in cap:
        return "Night"
    if "dusk" in cap or "dawn" in cap or "twilight" in cap:
        return "Twilight"
    return "Daylight"


def extract_ego_maneuver(caption: str, events: list) -> str:
    combined = (caption + " " + " ".join([e.get("event_description", "") for e in events])).lower()
    if "emergency brake" in combined or "hard brake" in combined:
        return "emergency brake"
    if "u-turn" in combined:
        return "U-turn"
    if "lane change left" in combined or "changes to the left lane" in combined:
        return "lane change left"
    if "lane change right" in combined or "changes to the right lane" in combined:
        return "lane change right"
    if "cut-in" in combined:
        return "cut-in"
    if "cut-out" in combined:
        return "cut-out"
    if "turn left" in combined or "turning left" in combined or "turns left" in combined:
        return "turn left"
    if "turn right" in combined or "turning right" in combined or "turns right" in combined:
        return "turn right"
    if "overtake" in combined or "passes" in combined:
        return "overtake"
    if "stopped" in combined or "halts" in combined or "comes to a stop" in combined:
        return "stop"
    if "following" in combined or "follows" in combined:
        return "follow"
    if "yield" in combined:
        return "yield"
    return "go straight"


def map_object_to_actor_type(obj_name: str) -> str:
    n = str(obj_name).lower()
    if "pedestrian" in n or "person" in n or "walker" in n:
        return "pedestrian"
    if "motorcyclist" in n or "motorcycle" in n or "scooter" in n:
        return "motorbike"
    if "bicycle" in n or "cyclist" in n or "bicyclist" in n:
        return "bicycle"
    if "bus" in n:
        return "bus"
    if "truck" in n or "semi" in n:
        return "truck"
    if "animal" in n or "dog" in n or "cat" in n:
        return "animal"
    return "car"


def extract_npc_actors(objects: list, caption: str, events: list) -> list[dict]:
    actors_map = {}
    for obj in objects:
        raw_name = obj.get("event", "Car")
        act_type = map_object_to_actor_type(raw_name)
        if act_type not in actors_map:
            # Determine likely maneuver
            cap_low = caption.lower()
            if act_type == "pedestrian" and ("crossing" in cap_low or "crosswalk" in cap_low):
                maneuver = "crossing road left to right"
            elif "parked" in cap_low and (act_type in ["truck", "car"]):
                maneuver = "stationary"
            else:
                maneuver = "follow"
            actors_map[act_type] = maneuver

    # Also check events for pedestrians/animals
    combined_evts = " ".join([e.get("event_description", "") for e in events]).lower()
    if "pedestrian" in combined_evts and "pedestrian" not in actors_map:
        maneuver = "crossing road left to right" if "cross" in combined_evts else "go straight"
        actors_map["pedestrian"] = maneuver
    if "animal" in combined_evts and "animal" not in actors_map:
        actors_map["animal"] = "crossing road left to right"

    if not actors_map:
        actors_map["car"] = "follow"

    return [{"actor_type": k, "maneuver": v} for k, v in actors_map.items()]


def map_event_class(raw_class: str) -> str:
    c = str(raw_class).lower()
    if "traffic control" in c:
        return "Traffic Control"
    if "vehicle interaction" in c:
        return "Vehicle Interaction"
    if "road and environment" in c or "environment" in c:
        return "Road and Environment"
    if "obstacle" in c:
        return "Obstacle Detection"
    if "animal" in c or "pedestrian" in c:
        return "Animal and Pedestrian"
    if "sudden" in c or "unexpected" in c:
        return "Sudden and Unexpected"
    return "Vehicle Interaction"


def map_event_pattern(raw_pattern: str) -> str:
    p = str(raw_pattern).lower()
    if "cross" in p:
        return "Cross-linear"
    if "causal" in p:
        return "Causal"
    if "predictive" in p or "likelihood" in p:
        return "Predictive"
    return "Co-linear"


def map_event_severity(raw_sev: str) -> str:
    s = str(raw_sev).lower()
    if "high" in s:
        return "High"
    if "medium" in s:
        return "medium"
    return "low"


def process_all_nuscenes():
    with open(NUSCENES_DIR / "nuscenes" / "Nuscenes_gemini_dataset_captions.json") as f:
        captions_list = json.load(f)
    with open(NUSCENES_DIR / "nuscenes" / "Nuscenes_gemini_events.json") as f:
        events_list = json.load(f)
    with open(NUSCENES_DIR / "nuscenes" / "nuscenes_gemini_object_timestamps.json") as f:
        objects_list = json.load(f)

    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    captions_map = {x["file_name"]: x["Caption"] for x in captions_list}
    events_map = {x["file_name"]: x.get("Events", []) for x in events_list}
    objects_map = {x["file_name"]: x.get("ObjectTimestamps", []) for x in objects_list}

    # Video paths mapping
    video_files = list(NUSCENES_DIR.rglob("*.mp4"))
    path_map = {vf.name: vf for vf in video_files}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing 1,000 NuScenes scenes into {OUTPUT_DIR}...")

    success_count = 0
    all_files = sorted(list(captions_map.keys()))

    for fname in all_files:
        video_id = Path(fname).stem
        caption = captions_map.get(fname, "")
        events = events_map.get(fname, [])
        objects = objects_map.get(fname, [])

        v_path = path_map.get(fname)
        rel_v_path = str(v_path.relative_to(BASE_DIR)) if v_path else f"videos/Nuscenes Dataset/{fname}"

        # Determine primary event
        if events:
            # Pick highest severity event or first event
            sorted_events = sorted(
                events,
                key=lambda e: 2 if "high" in e.get("event_severity", "").lower() else (1 if "medium" in e.get("event_severity", "").lower() else 0),
                reverse=True
            )
            prim_event = sorted_events[0]
            desc = prim_event.get("event_description", caption.split(".")[0])
            e_class = map_event_class(prim_event.get("event_type", "Vehicle Interaction"))
            e_pat = map_event_pattern(prim_event.get("event_pattern", "Co-linear"))
            e_sev = map_event_severity(prim_event.get("event_severity", "low"))
        else:
            desc = caption.split(".")[0] if caption else f"Driving scene {video_id}"
            e_class = "Vehicle Interaction"
            e_pat = "Co-linear"
            e_sev = "low"

        inter_ctrl, t_light = extract_traffic_light(caption, events)

        annotation = {
            "metadata": {
                "video_id": video_id,
                "dataset": "NuScenes",
                "video_path": rel_v_path
            },
            "road_network": {
                "road_type": extract_road_type(caption),
                "road_layout": extract_road_layout(caption, events),
                "no_of_lanes": extract_lanes(caption),
                "intersection_control": inter_ctrl,
                "traffic_sign": extract_traffic_signs(caption),
                "traffic_light": t_light
            },
            "environmental_conditions": {
                "weather": extract_weather(caption),
                "visibility": "Good",
                "lighting": extract_lighting(caption)
            },
            "ego_vehicle": {
                "maneuver": extract_ego_maneuver(caption, events)
            },
            "npc_actors": extract_npc_actors(objects, caption, events),
            "event_details": {
                "event_description": desc,
                "event_class": e_class,
                "event_pattern": e_pat,
                "event_severity": e_sev
            }
        }

        # Validate
        try:
            jsonschema.validate(instance=annotation, schema=schema)
        except jsonschema.ValidationError as err:
            print(f"Validation error for {video_id}: {err.message}")
            continue

        out_path = OUTPUT_DIR / f"{video_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)

        success_count += 1

    print(f"Successfully processed and validated {success_count}/1,000 NuScenes scenes in V2 schema!")


if __name__ == "__main__":
    process_all_nuscenes()

