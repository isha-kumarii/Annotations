# Video Scenario Annotation Project

This project contains the structure, schemas, reference documentation, and validation tooling for annotating driving scenarios, events, and behaviors from autonomous vehicle datasets (such as IDD).

---

## 📁 Folder Structure

```
Annotation/
├── videos/                  # 👈 Paste your video files here (e.g. IDD_0234.mp4)
├── annotations/             # 👈 JSON annotations corresponding to each video
│   └── example_IDD_0234.json # Reference example
├── schema/
│   ├── annotation_schema.json # Formal JSON Schema definition
│   └── template.json          # Starter blank template
├── docs/
│   ├── taxonomy.md            # Comprehensive taxonomy of events, patterns, maneuvers & actors
│   └── annotation_guidelines.md # Workflow and annotation instructions
├── scripts/
│   ├── generate_templates.py  # Auto-generates template JSON for newly added videos
│   └── validate_annotations.py# Validates all JSON annotations against the schema
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Paste your videos
Place your video files directly into the `videos/` folder:
```bash
# Example:
videos/IDD_0234.mp4
videos/IDD_0235.mp4
```

### 2. Generate starter annotation files
Run the automated template generator:
```bash
python3 scripts/generate_templates.py
```
This automatically scans `videos/` and creates corresponding `.json` files in `annotations/`.

### 3. Annotate
Open the generated JSON files in `annotations/` (e.g., `annotations/IDD_0234.json`) and fill in:
- **Environment**: road type, intersection, traffic control, weather, lighting.
- **Scenario Behavior**: event type, event pattern, severity, ego maneuver, key event description, and time window.
- **Actors**: class, count, and relative position.

Refer to [docs/taxonomy.md](file:///home/isha/Project/Annotation/docs/taxonomy.md) for full taxonomy definitions and allowed options.

### 4. Validate
Verify that all annotations conform to the schema:
```bash
python3 scripts/validate_annotations.py
```

---

## 📋 JSON Format Reference

```json
{
  "metadata": {
    "video_id": "IDD_0234",
    "dataset": "IDD"
  },
  "environment": {
    "road_type": "Urban",
    "intersection": true,
    "traffic_control": "Traffic signal",
    "weather": "Clear",
    "lighting": "Day"
  },
  "scenario_behavior": {
    "event_type": "Vehicle Interaction",
    "event_pattern": "Cross-linear",
    "severity": "Moderate",
    "ego_maneuver": "Go straight",
    "key_event": "Vehicle enters ego lane from right",
    "time_window": {
      "start_sec": "00:05",
      "end_sec": "00:12"
    }
  },
  "actors": [
    {
      "class": "car",
      "count": 1,
      "relative_position": "front_right"
    },
    {
      "class": "pedestrian",
      "count": 1,
      "relative_position": "sidewalk_left"
    }
  ]
}
```

