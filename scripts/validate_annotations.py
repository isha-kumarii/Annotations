#!/usr/bin/env python3
"""
validate_annotations.py

Validates all JSON files in the annotations/ folder against schema/annotation_schema.json.
Reports any syntax errors, missing fields, or invalid enum values.
"""

from pathlib import Path
import json
import sys

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

BASE_DIR = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = BASE_DIR / "annotations"
SCHEMA_PATH = BASE_DIR / "schema" / "annotation_schema.json"


def load_schema():
    if not SCHEMA_PATH.exists():
        print(f"Error: Schema file not found at {SCHEMA_PATH}")
        sys.exit(1)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_file(file_path, schema):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"JSON Syntax Error: {e}"

    if HAS_JSONSCHEMA:
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if errors:
            error_msgs = []
            for err in errors:
                loc = " -> ".join([str(p) for p in err.path]) if err.path else "root"
                error_msgs.append(f"  - [{loc}]: {err.message}")
            return False, "\n".join(error_msgs)
    else:
        # Basic check if jsonschema is absent
        for key in ["metadata", "environment", "scenario_behavior", "actors"]:
            if key not in data:
                return False, f"Missing required top-level key '{key}'"

    return True, "Valid"


def main():
    schema = load_schema()
    target_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ANNOTATIONS_DIR
    json_files = sorted(list(target_dir.rglob("*.json")))

    if not json_files:
        print(f"No JSON annotation files found in {target_dir}.")
        return

    print(f"Validating {len(json_files)} annotation file(s) in {target_dir.name} against schema...\n")
    all_passed = True

    for f in json_files:
        is_valid, msg = validate_file(f, schema)
        if is_valid:
            rel_f = f.relative_to(target_dir)
            print(f"✓ {rel_f}: OK")
        else:
            rel_f = f.relative_to(target_dir)
            print(f"✗ {rel_f}: FAILED")
            print(msg)
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("All annotations passed validation!")
    else:
        print("Validation errors detected. Please fix the issues listed above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

