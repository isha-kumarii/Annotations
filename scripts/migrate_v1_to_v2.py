#!/usr/bin/env python3
"""
migrate_v1_to_v2.py

Migrates V1 annotations (e.g. annotations_gemini/) into the enhanced V2 schema format (annotations_v2/).
Allows immediate testing of dataset catalog generation and hybrid retrieval.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "annotations_gemini"
OUTPUT_DIR = BASE_DIR / "annotations_v2"


def convert_maneuver(m):
    if not m:
        return "go straight"
    m_clean = str(m).strip().lower()
    if "left" in m_clean:
        return "turn left"
    if "right" in m_clean:
        return "turn right"
    if "stop" in m_clean:
        return "stop"
    if "follow" in m_clean:
        return "follow"
    if "overtake" in m_clean:
        return "overtake"
    if "yield" in m_clean:
        return "yield"
    if "brake" in m_clean:
        return "emergency brake"
    return "go straight"


def convert_severity(s):
    if not s:
        return "low"
    sc = str(s).strip().lower()
    if sc == "high":
        return "High"
    if sc in ["medium", "moderate"]:
        return "medium"
    return "low"


def convert_v1_to_v2(v1: dict, video_id: str, dataset: str) -> dict:
    env = v1.get("environment", {})
    sb = v1.get("scenario_behavior", {})
    actors = v1.get("actors", [])

    is_inter = bool(env.get("intersection", False))
    tc = env.get("traffic_control", "None")

    if is_inter:
        layout = "intersection"
        inter_ctrl = "signalized" if "signal" in tc.lower() else "unsignalized"
    else:
        layout = "straight road"
        inter_ctrl = "none"

    signs = ["stop"] if "stop" in tc.lower() else (["yield"] if "yield" in tc.lower() else [])
    light = "green" if "signal" in tc.lower() else "none"

    lighting_raw = str(env.get("lighting", "Day")).lower()
    if "night" in lighting_raw:
        lighting = "Night"
    elif "dawn" in lighting_raw or "dusk" in lighting_raw:
        lighting = "Twilight"
    else:
        lighting = "Daylight"

    npc_list = []
    for a in actors:
        cls_name = a.get("class", "car")
        pos = a.get("relative_position", "")
        if "crossing" in pos.lower():
            act_m = "crossing road left to right"
        else:
            act_m = "follow"
        npc_list.append({
            "actor_type": str(cls_name).lower(),
            "maneuver": act_m
        })
    if not npc_list:
        npc_list = [{"actor_type": "car", "maneuver": "follow"}]

    evt_class_raw = sb.get("event_type", "Vehicle Interaction")
    evt_class = evt_class_raw.replace("&", "and")

    evt_pat_raw = sb.get("event_pattern", "Co-linear")
    evt_pattern = "Predictive" if evt_pat_raw == "Likelihood" else evt_pat_raw

    v2 = {
        "metadata": {
            "video_id": video_id,
            "dataset": dataset,
            "video_path": f"videos/{dataset}/{video_id}.mp4"
        },
        "road_network": {
            "road_type": str(env.get("road_type", "urban")).lower(),
            "road_layout": layout,
            "no_of_lanes": 2,
            "intersection_control": inter_ctrl,
            "traffic_sign": signs,
            "traffic_light": light
        },
        "environmental_conditions": {
            "weather": str(env.get("weather", "clear")).lower(),
            "visibility": "Good",
            "lighting": lighting
        },
        "ego_vehicle": {
            "maneuver": convert_maneuver(sb.get("ego_maneuver"))
        },
        "npc_actors": npc_list,
        "event_details": {
            "event_description": sb.get("key_event", f"Driving event in {video_id}"),
            "event_class": evt_class,
            "event_pattern": evt_pattern,
            "event_severity": convert_severity(sb.get("severity"))
        }
    }
    return v2


def main():
    json_files = list(INPUT_DIR.rglob("*.json"))
    print(f"Found {len(json_files)} files in {INPUT_DIR}")
    converted = 0
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                d = json.load(f)
            vid = jf.stem
            ds = jf.parent.name
            v2_data = convert_v1_to_v2(d, vid, ds)

            out_p = OUTPUT_DIR / ds / f"{vid}.json"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(v2_data, f, indent=2)
            converted += 1
        except Exception as e:
            print(f"Error converting {jf.name}: {e}")

    print(f"Successfully converted {converted} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

