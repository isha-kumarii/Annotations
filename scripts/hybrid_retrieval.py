#!/usr/bin/env python3
"""
hybrid_retrieval.py

Dual-scoring hybrid retrieval engine for autonomous driving video scenarios:
  1. Semantic Similarity Score (S_semantic) between query text & scenario descriptions.
  2. Structured JSON Comparison Score (S_structured) across schema dimensions:
     - Road Network
     - Environmental Conditions
     - Ego Vehicle Maneuver
     - NPC Actors & Maneuvers
     - Event Details (Class, Pattern, Severity)
  3. Total Hybrid Score: S_total = α * S_semantic + β * S_structured
  4. Returns & displays the Top-K ranked videos.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = BASE_DIR / "dataset_catalog.json"


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    cleaned = "".join([c.lower() if c.isalnum() else " " for c in text])
    return [w for w in cleaned.split() if len(w) > 1]


def compute_tf_idf_similarity(query_tokens: list[str], doc_tokens: list[str], idf_dict: dict) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    q_counts = Counter(query_tokens)
    d_counts = Counter(doc_tokens)

    dot_product = 0.0
    q_norm = 0.0
    d_norm = 0.0

    all_terms = set(q_counts.keys()) | set(d_counts.keys())
    for t in all_terms:
        idf = idf_dict.get(t, 1.0)
        q_val = (q_counts.get(t, 0) / len(query_tokens)) * idf
        d_val = (d_counts.get(t, 0) / len(doc_tokens)) * idf

        dot_product += q_val * d_val
        q_norm += q_val ** 2
        d_norm += d_val ** 2

    if q_norm == 0.0 or d_norm == 0.0:
        return 0.0
    return dot_product / (math.sqrt(q_norm) * math.sqrt(d_norm))


def compare_structured_json(query_ann: dict, cand_ann: dict) -> tuple[float, list[str]]:
    """
    Computes structured alignment score between query JSON and candidate JSON.
    Returns: (score between 0 and 1, list of matched tags)
    """
    matches = []
    total_weights = 0.0
    earned_score = 0.0

    # 1. Road Network
    q_rn = query_ann.get("road_network", {})
    c_rn = cand_ann.get("road_network", {})

    if "road_type" in q_rn and q_rn["road_type"]:
        total_weights += 1.0
        if q_rn["road_type"].lower() == c_rn.get("road_type", "").lower():
            earned_score += 1.0
            matches.append(f"Road Type: {q_rn['road_type']} ✓")

    if "road_layout" in q_rn and q_rn["road_layout"]:
        total_weights += 1.0
        if q_rn["road_layout"].lower() == c_rn.get("road_layout", "").lower():
            earned_score += 1.0
            matches.append(f"Layout: {q_rn['road_layout']} ✓")

    if "intersection_control" in q_rn and q_rn["intersection_control"] != "none":
        total_weights += 0.8
        if q_rn["intersection_control"].lower() == c_rn.get("intersection_control", "").lower():
            earned_score += 0.8
            matches.append(f"Inter. Control: {q_rn['intersection_control']} ✓")

    if "traffic_light" in q_rn and q_rn["traffic_light"] != "none":
        total_weights += 0.8
        if q_rn["traffic_light"].lower() == c_rn.get("traffic_light", "").lower():
            earned_score += 0.8
            matches.append(f"Signal: {q_rn['traffic_light']} ✓")

    # 2. Environmental Conditions
    q_env = query_ann.get("environmental_conditions", {})
    c_env = cand_ann.get("environmental_conditions", {})

    if "weather" in q_env and q_env["weather"]:
        total_weights += 1.2
        if q_env["weather"].lower() == c_env.get("weather", "").lower():
            earned_score += 1.2
            matches.append(f"Weather: {q_env['weather']} ✓")

    if "lighting" in q_env and q_env["lighting"] != "Unknown":
        total_weights += 0.8
        if q_env["lighting"].lower() == c_env.get("lighting", "").lower():
            earned_score += 0.8
            matches.append(f"Lighting: {q_env['lighting']} ✓")

    # 3. Ego Vehicle Maneuver
    q_ego = query_ann.get("ego_vehicle", {})
    c_ego = cand_ann.get("ego_vehicle", {})

    if "maneuver" in q_ego and q_ego["maneuver"]:
        total_weights += 1.5
        if q_ego["maneuver"].lower() == c_ego.get("maneuver", "").lower():
            earned_score += 1.5
            matches.append(f"Ego Maneuver: {q_ego['maneuver']} ✓")

    # 4. NPC Actors
    q_actors = query_ann.get("npc_actors", [])
    c_actors = cand_ann.get("npc_actors", [])

    if q_actors:
        total_weights += 2.0
        q_act_set = set(a.get("actor_type", "").lower() for a in q_actors if a.get("actor_type"))
        c_act_set = set(a.get("actor_type", "").lower() for a in c_actors if a.get("actor_type"))

        if q_act_set and c_act_set:
            overlap = q_act_set & c_act_set
            jaccard = len(overlap) / len(q_act_set | c_act_set)
            earned_score += 2.0 * jaccard
            if overlap:
                matches.append(f"Actors: {', '.join(sorted(list(overlap)))} ✓")

    # 5. Event Details
    q_evt = query_ann.get("event_details", {})
    c_evt = cand_ann.get("event_details", {})

    if "event_class" in q_evt and q_evt["event_class"]:
        total_weights += 1.5
        if q_evt["event_class"].lower() == c_evt.get("event_class", "").lower():
            earned_score += 1.5
            matches.append(f"Class: {q_evt['event_class']} ✓")

    if "event_pattern" in q_evt and q_evt["event_pattern"]:
        total_weights += 1.0
        if q_evt["event_pattern"].lower() == c_evt.get("event_pattern", "").lower():
            earned_score += 1.0
            matches.append(f"Pattern: {q_evt['event_pattern']} ✓")

    if "event_severity" in q_evt and q_evt["event_severity"]:
        total_weights += 0.8
        qs = q_evt["event_severity"].lower()
        cs = c_evt.get("event_severity", "").lower()
        if qs == cs:
            earned_score += 0.8
            matches.append(f"Severity: {q_evt['event_severity']} ✓")

    score = (earned_score / total_weights) if total_weights > 0 else 0.5
    return round(score, 4), matches


def run_hybrid_search(
    query_json: dict,
    catalog: dict,
    alpha: float = 0.5,
    beta: float = 0.5,
    top_k: int = 5
) -> list[dict]:
    videos = catalog.get("videos", [])
    if not videos:
        return []

    # Build corpus IDF
    doc_count = len(videos)
    doc_freq = Counter()
    for v in videos:
        doc_tokens = set(tokenize(v["search_index"]["text_document"]))
        for t in doc_tokens:
            doc_freq[t] += 1

    idf_dict = {
        t: math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
        for t, df in doc_freq.items()
    }

    # Query text
    q_text = (
        query_json.get("event_details", {}).get("event_description", "") or
        query_json.get("query_text", "")
    )
    q_tokens = tokenize(q_text)

    ranked_results = []
    for cand in videos:
        # 1. Semantic score
        c_tokens = tokenize(cand["search_index"]["text_document"])
        s_sem = compute_tf_idf_similarity(q_tokens, c_tokens, idf_dict) if q_tokens else 0.5

        # 2. Structured score
        s_struct, matches = compare_structured_json(query_json, cand["annotation"])

        # 3. Hybrid score
        s_total = alpha * s_sem + beta * s_struct

        ranked_results.append({
            "video_id": cand["video_id"],
            "dataset": cand["dataset"],
            "video_path": cand["video_path"],
            "event_description": cand["search_index"]["event_description"],
            "score_total": round(s_total, 4),
            "score_semantic": round(s_sem, 4),
            "score_structured": round(s_struct, 4),
            "matches": matches,
            "annotation": cand["annotation"]
        })

    ranked_results.sort(key=lambda x: x["score_total"], reverse=True)
    return ranked_results[:top_k]


def display_results(results: list[dict], query_desc: str):
    print("\n" + "=" * 90)
    print(f" TOP {len(results)} HYBRID RETRIEVAL RESULTS ".center(90, "="))
    print(f" Query: {query_desc} ".center(90))
    print("=" * 90)

    for rank, r in enumerate(results, start=1):
        print(f"\n[Rank #{rank}] {r['video_id']} ({r['dataset'].upper()})")
        print(f"  Score: {r['score_total']:.4f}  [Semantic: {r['score_semantic']:.4f} | Structured: {r['score_structured']:.4f}]")
        print(f"  Path: {r['video_path']}")
        print(f"  Event: {r['event_description']}")
        if r['matches']:
            print(f"  Matched Attributes: {' | '.join(r['matches'])}")
        print("-" * 90)


def main():
    parser = argparse.ArgumentParser(description="Hybrid retrieval: Semantic Similarity + Structured JSON Comparison.")
    parser.add_argument("--catalog", type=str, default=str(DEFAULT_CATALOG_PATH), help="Path to dataset_catalog.json")
    parser.add_argument("--query-json", type=str, help="Path to query JSON file.")
    parser.add_argument("--query-text", type=str, help="Natural language query description.")
    parser.add_argument("--weather", type=str, help="Filter/match weather (e.g. rain, clear).")
    parser.add_argument("--road-type", type=str, help="Filter/match road type (e.g. urban, highway).")
    parser.add_argument("--event-class", type=str, help="Filter/match event class (e.g. 'Animal and Pedestrian').")
    parser.add_argument("--actor", type=str, help="Filter/match actor type (e.g. pedestrian, motorcycle).")
    parser.add_argument("--maneuver", type=str, help="Filter/match ego maneuver (e.g. 'turn left', follow).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic similarity (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for structured comparison (default: 0.5)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve (default: 5)")
    args = parser.parse_args()

    # Load catalog
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"Error: Dataset catalog {catalog_path} not found. Run scripts/build_dataset_json.py first.")
        sys.exit(1)

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Build query JSON
    query_obj = {}
    if args.query_json:
        q_path = Path(args.query_json)
        if q_path.is_file():
            with open(q_path, "r", encoding="utf-8") as f:
                query_obj = json.load(f)
        else:
            print(f"Error: Query JSON file not found: {args.query_json}")
            sys.exit(1)

    # Apply CLI overrides if passed
    if args.query_text:
        query_obj.setdefault("event_details", {})["event_description"] = args.query_text
    if args.weather:
        query_obj.setdefault("environmental_conditions", {})["weather"] = args.weather
    if args.road_type:
        query_obj.setdefault("road_network", {})["road_type"] = args.road_type
    if args.event_class:
        query_obj.setdefault("event_details", {})["event_class"] = args.event_class
    if args.actor:
        query_obj.setdefault("npc_actors", []).append({"actor_type": args.actor, "maneuver": "follow"})
    if args.maneuver:
        query_obj.setdefault("ego_vehicle", {})["maneuver"] = args.maneuver

    if not query_obj:
        # Default sample query
        query_obj = {
            "environmental_conditions": {"weather": "rain"},
            "npc_actors": [{"actor_type": "pedestrian", "maneuver": "crossing road left to right"}],
            "event_details": {
                "event_description": "Pedestrian crossing road on wet rainy asphalt",
                "event_class": "Animal and Pedestrian"
            }
        }

    desc = query_obj.get("event_details", {}).get("event_description", "Custom Query")
    results = run_hybrid_search(
        query_json=query_obj,
        catalog=catalog,
        alpha=args.alpha,
        beta=args.beta,
        top_k=args.top_k
    )

    display_results(results, desc)


if __name__ == "__main__":
    main()

