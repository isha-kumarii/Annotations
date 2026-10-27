# Annotation Guidelines & Workflow

## Project Structure Overview

```
Annotation/
├── videos/                  <-- Paste your raw video files here (.mp4, .avi, .mkv, etc.)
├── annotations/             <-- Store the resulting JSON annotation files here
├── schema/
│   ├── annotation_schema.json   <-- JSON Schema validating field structure and enum values
│   └── template.json            <-- Clean starter template for new annotations
├── docs/
│   ├── taxonomy.md              <-- Full taxonomy definitions and allowed values
│   └── annotation_guidelines.md <-- Instructions and workflow guide
├── scripts/
│   ├── generate_templates.py    <-- Auto-creates blank JSON files in annotations/ for videos
│   └── validate_annotations.py  <-- Checks all JSON annotations for schema compliance
└── README.md
```

## Annotation Step-by-Step Workflow

1. **Add Videos**:
   Paste video files into the `videos/` folder.
   Recommended naming convention: `<DATASET>_<ID>.<ext>`, e.g., `IDD_0234.mp4`.

2. **Generate / Create Annotation Files**:
   Each video has a corresponding JSON file in `annotations/` sharing the same base name:
   - Video: `videos/IDD_0234.mp4`
   - Annotation: `annotations/IDD_0234.json`

   You can manually copy `schema/template.json` to `annotations/<video_id>.json`, or run:
   ```bash
   python3 scripts/generate_templates.py
   ```
   This script will detect any video files in `videos/` that lack an annotation and create a blank template automatically.

3. **Fill Out the Annotation**:
   - Set `metadata.video_id` to match the video name (e.g., `IDD_0234`).
   - Populate `environment` fields (`road_type`, `intersection`, `traffic_control`, `weather`, `lighting`).
   - Identify the primary event under `scenario_behavior` (`event_type`, `event_pattern`, `severity`, `ego_maneuver`, `key_event`, and `time_window`).
   - Add all interacting agents to `actors`.

4. **Validate Your Annotations**:
   Run the validator script to ensure there are no syntax errors or invalid enum values:
   ```bash
   python3 scripts/validate_annotations.py
   ```

