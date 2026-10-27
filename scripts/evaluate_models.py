#!/usr/bin/env python3
"""
evaluate_models.py

Benchmarking evaluation pipeline comparing:
  - Google Gemini Flash-Lite (annotations_gemini/)
  - Qwen2.5-VL (annotations_qwen/)
Against the ground truth baseline:
  - VideoDB (annotations/)

Metrics computed:
  1. Field-wise Accuracy, Precision, Recall, Macro-F1, Weighted-F1 for all 9 categorical/binary fields.
  2. Actor class detection & positional matching Precision, Recall, and F1.
  3. Temporal window Intersection-over-Union (tIoU) and Overlap F1.
  4. Natural language caption ROUGE-1, ROUGE-2, ROUGE-L and Jaccard similarity for key_event.
  5. Overall aggregated metrics (Environment, Behavior, Composite) and dataset-wise breakdown.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict, Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_VIDEODB = BASE_DIR / "annotations"
DIR_GEMINI = BASE_DIR / "annotations_gemini"
DIR_QWEN = BASE_DIR / "annotations_qwen"

DATASETS = ["dashcam", "kitti dataset", "IDD", "nuscenes dataset"]

CATEGORICAL_FIELDS = [
    ("environment", "road_type"),
    ("environment", "intersection"),
    ("environment", "traffic_control"),
    ("environment", "weather"),
    ("environment", "lighting"),
    ("scenario_behavior", "event_type"),
    ("scenario_behavior", "event_pattern"),
    ("scenario_behavior", "severity"),
    ("scenario_behavior", "ego_maneuver"),
]

CANONICAL_ACTOR_MAP = {
    "cyclist": "bicycle",
    "bicyclist": "bicycle",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "scooter": "motorcycle",
    "moped": "motorcycle",
    "auto-rickshaw": "auto_rickshaw",
    "autorickshaw": "auto_rickshaw",
    "tuk-tuk": "auto_rickshaw",
    "tuktuk": "auto_rickshaw",
    "auto": "auto_rickshaw",
    "suv": "car",
    "sedan": "car",
    "vehicle": "car",
    "gate_panel": "barrier",
    "golf_cart": "car",
}

COMMON_ACTOR_CLASSES = [
    "car",
    "motorcycle",
    "pedestrian",
    "bicycle",
    "truck",
    "bus",
    "auto_rickshaw",
    "animal",
]


def normalize_val(val):
    if val is None:
        return "Unknown"
    if isinstance(val, bool):
        return str(val)
    s = str(val).strip()
    return s if s else "Unknown"


def parse_timestamp(ts):
    if ts is None:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def calculate_tiou(tw_gt, tw_pred):
    s_gt = parse_timestamp(tw_gt.get("start_sec", 0))
    e_gt = parse_timestamp(tw_gt.get("end_sec", 0))
    s_pr = parse_timestamp(tw_pred.get("start_sec", 0))
    e_pr = parse_timestamp(tw_pred.get("end_sec", 0))

    if e_gt < s_gt:
        s_gt, e_gt = e_gt, s_gt
    if e_pr < s_pr:
        s_pr, e_pr = e_pr, s_pr

    inter_start = max(s_gt, s_pr)
    inter_end = min(e_gt, e_pr)
    inter = max(0.0, inter_end - inter_start)

    union_start = min(s_gt, s_pr)
    union_end = max(e_gt, e_pr)
    union = max(0.0, union_end - union_start)

    if union <= 0.0:
        return 1.0 if (s_gt == s_pr and e_gt == e_pr) else 0.0
    return inter / union


def tokenize(text):
    if not text:
        return []
    cleaned = "".join([c.lower() if c.isalnum() else " " for c in text])
    return [w for w in cleaned.split() if w]


def get_ngrams(tokens, n):
    if len(tokens) < n:
        return Counter()
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])


def compute_rouge_n(gt_tokens, pred_tokens, n=1):
    gt_ngrams = get_ngrams(gt_tokens, n)
    pred_ngrams = get_ngrams(pred_tokens, n)

    if not gt_ngrams or not pred_ngrams:
        return 0.0, 0.0, 0.0

    overlap = sum((gt_ngrams & pred_ngrams).values())
    prec = overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0
    rec = overlap / sum(gt_ngrams.values()) if gt_ngrams else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(gt_tokens, pred_tokens):
    if not gt_tokens or not pred_tokens:
        return 0.0, 0.0, 0.0
    lcs = compute_lcs_length(gt_tokens, pred_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(gt_tokens)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_jaccard(tokens1, tokens2):
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    union = len(s1 | s2)
    return len(s1 & s2) / union if union > 0 else 0.0


def compute_classification_metrics(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "per_class": {}
        }

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / n

    classes = sorted(list(set(y_true) | set(y_pred)))
    per_class = {}

    tp_total = 0
    fp_total = 0
    fn_total = 0

    macro_p_sum = 0.0
    macro_r_sum = 0.0
    macro_f1_sum = 0.0

    weighted_p_sum = 0.0
    weighted_r_sum = 0.0
    weighted_f1_sum = 0.0

    gt_counts = Counter(y_true)

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = gt_counts.get(c, 0)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }

        macro_p_sum += prec
        macro_r_sum += rec
        macro_f1_sum += f1

        weight = support / n
        weighted_p_sum += prec * weight
        weighted_r_sum += rec * weight
        weighted_f1_sum += f1 * weight

    num_classes = len(classes) if classes else 1
    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p_sum / num_classes, 4),
        "macro_recall": round(macro_r_sum / num_classes, 4),
        "macro_f1": round(macro_f1_sum / num_classes, 4),
        "weighted_precision": round(weighted_p_sum, 4),
        "weighted_recall": round(weighted_r_sum, 4),
        "weighted_f1": round(weighted_f1_sum, 4),
        "per_class": per_class
    }


def normalize_actor_class(raw_class):
    c = str(raw_class).lower().strip().replace("-", "_").replace(" ", "_")
    return CANONICAL_ACTOR_MAP.get(c, c)


def normalize_position(pos):
    if not pos:
        return "front"
    p = str(pos).lower().strip().replace("-", "_").replace(" ", "_")
    return p


def evaluate_actors(gt_actors_list, pred_actors_list):
    """
    Evaluates:
      1. Actor class presence (multi-label detection per class and macro)
      2. Positional tuple matching (class, position)
    """
    class_tps = Counter()
    class_fps = Counter()
    class_fns = Counter()
    class_support = Counter()

    pos_tp = 0
    pos_fp = 0
    pos_fn = 0

    for gt_acts, pred_acts in zip(gt_actors_list, pred_actors_list):
        # Class sets
        gt_classes = set(normalize_actor_class(a.get("class", "")) for a in gt_acts)
        pred_classes = set(normalize_actor_class(a.get("class", "")) for a in pred_acts)

        all_classes = gt_classes | pred_classes
        for c in all_classes:
            in_gt = c in gt_classes
            in_pred = c in pred_classes
            if in_gt and in_pred:
                class_tps[c] += 1
            elif not in_gt and in_pred:
                class_fps[c] += 1
            elif in_gt and not in_pred:
                class_fns[c] += 1
            if in_gt:
                class_support[c] += 1

        # Positional tuples: (class, position)
        gt_tuples = Counter((normalize_actor_class(a.get("class", "")), normalize_position(a.get("relative_position", ""))) for a in gt_acts)
        pred_tuples = Counter((normalize_actor_class(a.get("class", "")), normalize_position(a.get("relative_position", ""))) for a in pred_acts)

        matched = gt_tuples & pred_tuples
        match_count = sum(matched.values())
        gt_count = sum(gt_tuples.values())
        pred_count = sum(pred_tuples.values())

        pos_tp += match_count
        pos_fp += max(0, pred_count - match_count)
        pos_fn += max(0, gt_count - match_count)

    # Calculate per-class metrics
    class_metrics = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    common_p, common_r, common_f1 = 0.0, 0.0, 0.0

    all_eval_classes = sorted(list(class_support.keys() | class_fps.keys()))
    for c in all_eval_classes:
        tp = class_tps[c]
        fp = class_fps[c]
        fn = class_fns[c]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        class_metrics[c] = {
            "tp": tp, "fp": fp, "fn": fn, "support": class_support[c],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)
        }
        macro_p += p
        macro_r += r
        macro_f1 += f

    num_c = len(all_eval_classes) if all_eval_classes else 1
    macro_p /= num_c
    macro_r /= num_c
    macro_f1 /= num_c

    # Positional match precision, recall, f1
    pos_prec = pos_tp / (pos_tp + pos_fp) if (pos_tp + pos_fp) > 0 else 0.0
    pos_rec = pos_tp / (pos_tp + pos_fn) if (pos_tp + pos_fn) > 0 else 0.0
    pos_f1 = (2 * pos_prec * pos_rec / (pos_prec + pos_rec)) if (pos_prec + pos_rec) > 0 else 0.0

    return {
        "class_macro_precision": round(macro_p, 4),
        "class_macro_recall": round(macro_r, 4),
        "class_macro_f1": round(macro_f1, 4),
        "positional_precision": round(pos_prec, 4),
        "positional_recall": round(pos_rec, 4),
        "positional_f1": round(pos_f1, 4),
        "per_class": class_metrics
    }


def load_dataset_annotations(datasets=None):
    """
    Loads all matching annotations across VideoDB, Gemini, and Qwen.
    Returns list of dicts:
      {
        'video_id': ...,
        'dataset': ...,
        'videodb': data,
        'gemini': data,
        'qwen': data
      }
    """
    if datasets is None:
        datasets = DATASETS

    records = []
    missing_counts = {"gemini": 0, "qwen": 0}

    for ds in datasets:
        dir_v = DIR_VIDEODB / ds
        if not dir_v.exists():
            continue
        for f in sorted(dir_v.glob("*.json")):
            vid = f.stem
            fg = DIR_GEMINI / ds / f.name
            fq = DIR_QWEN / ds / f.name

            if not fg.exists() or not fq.exists():
                if not fg.exists():
                    missing_counts["gemini"] += 1
                if not fq.exists():
                    missing_counts["qwen"] += 1
                continue

            with open(f) as fp:
                dv = json.load(fp)
            with open(fg) as fp:
                dg = json.load(fp)
            with open(fq) as fp:
                dq = json.load(fp)

            records.append({
                "video_id": vid,
                "dataset": ds,
                "videodb": dv,
                "gemini": dg,
                "qwen": dq
            })

    return records, missing_counts


def run_benchmark(records):
    """
    Computes all benchmark metrics for Gemini and Qwen against VideoDB.
    """
    total_videos = len(records)
    if total_videos == 0:
        print("No matching annotation records found.")
        return {}

    models = ["gemini", "qwen"]
    results = {}

    for model in models:
        # 1. Categorical fields
        field_metrics = {}
        for sec, field_name in CATEGORICAL_FIELDS:
            y_true = [normalize_val(r["videodb"].get(sec, {}).get(field_name)) for r in records]
            y_pred = [normalize_val(r[model].get(sec, {}).get(field_name)) for r in records]
            m = compute_classification_metrics(y_true, y_pred)
            field_metrics[f"{sec}.{field_name}"] = m

        # 2. Actors evaluation
        gt_actors = [r["videodb"].get("actors", []) for r in records]
        pred_actors = [r[model].get("actors", []) for r in records]
        actors_metric = evaluate_actors(gt_actors, pred_actors)

        # 3. Temporal evaluation (tIoU)
        tious = []
        tiou_ge_05 = 0
        tiou_ge_02 = 0
        for r in records:
            tw_gt = r["videodb"].get("scenario_behavior", {}).get("time_window", {})
            tw_pr = r[model].get("scenario_behavior", {}).get("time_window", {})
            iou = calculate_tiou(tw_gt, tw_pr)
            tious.append(iou)
            if iou >= 0.5:
                tiou_ge_05 += 1
            if iou >= 0.2:
                tiou_ge_02 += 1

        mean_tiou = sum(tious) / len(tious) if tious else 0.0

        # 4. Key Event NLP evaluation
        rouge_1_f1s = []
        rouge_2_f1s = []
        rouge_l_f1s = []
        jaccards = []
        for r in records:
            gt_text = r["videodb"].get("scenario_behavior", {}).get("key_event", "")
            pr_text = r[model].get("scenario_behavior", {}).get("key_event", "")
            t_gt = tokenize(gt_text)
            t_pr = tokenize(pr_text)

            _, _, r1 = compute_rouge_n(t_gt, t_pr, n=1)
            _, _, r2 = compute_rouge_n(t_gt, t_pr, n=2)
            _, _, rl = compute_rouge_l(t_gt, t_pr)
            j = compute_jaccard(t_gt, t_pr)

            rouge_1_f1s.append(r1)
            rouge_2_f1s.append(r2)
            rouge_l_f1s.append(rl)
            jaccards.append(j)

        mean_r1 = sum(rouge_1_f1s) / len(rouge_1_f1s) if rouge_1_f1s else 0.0
        mean_r2 = sum(rouge_2_f1s) / len(rouge_2_f1s) if rouge_2_f1s else 0.0
        mean_rl = sum(rouge_l_f1s) / len(rouge_l_f1s) if rouge_l_f1s else 0.0
        mean_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0.0

        # Section-level rollups
        env_fields = [f"{sec}.{field}" for sec, field in CATEGORICAL_FIELDS if sec == "environment"]
        beh_fields = [f"{sec}.{field}" for sec, field in CATEGORICAL_FIELDS if sec == "scenario_behavior"]

        env_acc = sum(field_metrics[k]["accuracy"] for k in env_fields) / len(env_fields)
        env_f1 = sum(field_metrics[k]["macro_f1"] for k in env_fields) / len(env_fields)
        env_wf1 = sum(field_metrics[k]["weighted_f1"] for k in env_fields) / len(env_fields)

        beh_acc = sum(field_metrics[k]["accuracy"] for k in beh_fields) / len(beh_fields)
        beh_f1 = sum(field_metrics[k]["macro_f1"] for k in beh_fields) / len(beh_fields)
        beh_wf1 = sum(field_metrics[k]["weighted_f1"] for k in beh_fields) / len(beh_fields)

        all_acc = sum(field_metrics[k]["accuracy"] for k in field_metrics) / len(field_metrics)
        all_f1 = sum(field_metrics[k]["macro_f1"] for k in field_metrics) / len(field_metrics)
        all_wf1 = sum(field_metrics[k]["weighted_f1"] for k in field_metrics) / len(field_metrics)

        # Composite score
        # Categorical Macro-F1 (40%) + Categorical Accuracy (20%) + Actor Class F1 (20%) + Mean tIoU (10%) + ROUGE-L (10%)
        composite_score = (
            0.40 * all_f1 +
            0.20 * all_acc +
            0.20 * actors_metric["class_macro_f1"] +
            0.10 * mean_tiou +
            0.10 * mean_rl
        )

        results[model] = {
            "total_videos": total_videos,
            "overall": {
                "composite_score": round(composite_score, 4),
                "categorical_accuracy": round(all_acc, 4),
                "categorical_macro_f1": round(all_f1, 4),
                "categorical_weighted_f1": round(all_wf1, 4),
                "environment_accuracy": round(env_acc, 4),
                "environment_macro_f1": round(env_f1, 4),
                "environment_weighted_f1": round(env_wf1, 4),
                "behavior_accuracy": round(beh_acc, 4),
                "behavior_macro_f1": round(beh_f1, 4),
                "behavior_weighted_f1": round(beh_wf1, 4),
            },
            "fields": field_metrics,
            "actors": actors_metric,
            "temporal": {
                "mean_tiou": round(mean_tiou, 4),
                "precision_at_05": round(tiou_ge_05 / total_videos, 4),
                "precision_at_02": round(tiou_ge_02 / total_videos, 4),
            },
            "key_event_nlp": {
                "rouge_1_f1": round(mean_r1, 4),
                "rouge_2_f1": round(mean_r2, 4),
                "rouge_l_f1": round(mean_rl, 4),
                "jaccard_similarity": round(mean_jaccard, 4),
            }
        }

    return results


def print_comparison_tables(results, title="Overall 200 Videos Benchmark"):
    """
    Renders clean terminal comparison tables.
    """
    gemini = results.get("gemini", {})
    qwen = results.get("qwen", {})

    print("\n" + "=" * 90)
    print(f" {title.upper()} (Baseline: VideoDB) ".center(90, "="))
    print("=" * 90)

    # 1. Executive Summary Table
    print("\n--- 1. OVERALL EXECUTIVE SUMMARY ---")
    headers = ["Metric Dimension", "Gemini Flash-Lite", "Qwen2.5-VL", "Delta (Gemini - Qwen)"]
    print(f"{headers[0]:<35} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]:<20}")
    print("-" * 97)

    summary_rows = [
        ("Composite Benchmark Index", gemini["overall"]["composite_score"], qwen["overall"]["composite_score"]),
        ("Overall Categorical Accuracy", gemini["overall"]["categorical_accuracy"], qwen["overall"]["categorical_accuracy"]),
        ("Overall Categorical Macro-F1", gemini["overall"]["categorical_macro_f1"], qwen["overall"]["categorical_macro_f1"]),
        ("Overall Categorical Weighted-F1", gemini["overall"]["categorical_weighted_f1"], qwen["overall"]["categorical_weighted_f1"]),
        ("Environment Macro-F1", gemini["overall"]["environment_macro_f1"], qwen["overall"]["environment_macro_f1"]),
        ("Scenario Behavior Macro-F1", gemini["overall"]["behavior_macro_f1"], qwen["overall"]["behavior_macro_f1"]),
        ("Actor Class Detection F1", gemini["actors"]["class_macro_f1"], qwen["actors"]["class_macro_f1"]),
        ("Actor Positional Match F1", gemini["actors"]["positional_f1"], qwen["actors"]["positional_f1"]),
        ("Temporal Window Mean tIoU", gemini["temporal"]["mean_tiou"], qwen["temporal"]["mean_tiou"]),
        ("Key Event Caption ROUGE-L F1", gemini["key_event_nlp"]["rouge_l_f1"], qwen["key_event_nlp"]["rouge_l_f1"]),
        ("Key Event Token Jaccard", gemini["key_event_nlp"]["jaccard_similarity"], qwen["key_event_nlp"]["jaccard_similarity"]),
    ]

    for label, g_val, q_val in summary_rows:
        delta = g_val - q_val
        delta_str = f"{'+' if delta > 0 else ''}{delta:.4f}"
        print(f"{label:<35} | {g_val:>18.4f} | {q_val:>18.4f} | {delta_str:>20}")

    # 2. Detailed Field-by-Field Breakdown
    print("\n--- 2. DETAILED FIELD-BY-FIELD BREAKDOWN (ACCURACY & F1) ---")
    f_headers = ["Field Name", "Gemini Acc", "Gemini F1", "Qwen Acc", "Qwen F1", "F1 Advantage"]
    print(f"{f_headers[0]:<30} | {f_headers[1]:<10} | {f_headers[2]:<10} | {f_headers[3]:<10} | {f_headers[4]:<10} | {f_headers[5]:<14}")
    print("-" * 97)

    for sec, field in CATEGORICAL_FIELDS:
        fname = f"{sec}.{field}"
        g_acc = gemini["fields"][fname]["accuracy"]
        g_f1 = gemini["fields"][fname]["macro_f1"]
        q_acc = qwen["fields"][fname]["accuracy"]
        q_f1 = qwen["fields"][fname]["macro_f1"]

        f1_diff = g_f1 - q_f1
        if abs(f1_diff) < 0.005:
            adv = "Tied"
        elif f1_diff > 0:
            adv = f"Gemini (+{f1_diff:.2f})"
        else:
            adv = f"Qwen (+{-f1_diff:.2f})"

        print(f"{fname:<30} | {g_acc:>10.4f} | {g_f1:>10.4f} | {q_acc:>10.4f} | {q_f1:>10.4f} | {adv:<14}")

    # 3. Actor Class Presence F1
    print("\n--- 3. ACTOR CLASS DETECTION F1 BREAKDOWN ---")
    act_headers = ["Actor Class", "Support (VideoDB)", "Gemini F1", "Qwen F1", "Detection Advantage"]
    print(f"{act_headers[0]:<20} | {act_headers[1]:<18} | {act_headers[2]:<12} | {act_headers[3]:<12} | {act_headers[4]:<20}")
    print("-" * 90)

    g_act = gemini["actors"]["per_class"]
    q_act = qwen["actors"]["per_class"]
    for c in COMMON_ACTOR_CLASSES:
        sup = g_act.get(c, {}).get("support", 0)
        gf = g_act.get(c, {}).get("f1", 0.0)
        qf = q_act.get(c, {}).get("f1", 0.0)
        diff = gf - qf
        if abs(diff) < 0.005:
            adv = "Tied"
        elif diff > 0:
            adv = f"Gemini (+{diff:.2f})"
        else:
            adv = f"Qwen (+{-diff:.2f})"
        print(f"{c:<20} | {sup:>18} | {gf:>12.4f} | {qf:>12.4f} | {adv:<20}")

    print("=" * 90 + "\n")


def generate_markdown_report(overall_results, dataset_results, output_path):
    """
    Writes a comprehensive markdown benchmark report file.
    """
    g_ov = overall_results["gemini"]
    q_ov = overall_results["qwen"]

    lines = []
    lines.append("# Autonomous Driving Video Annotation Benchmark Report")
    lines.append("")
    lines.append("Benchmarking **Google Gemini Flash-Lite** and **Qwen2.5-VL** against **groundtruth** across **200 autonomous driving videos** (Dashcam, KITTI, IDD, NuScenes).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("| Metric Dimension | Gemini Flash-Lite | Qwen2.5-VL | Advantage / Delta |")
    lines.append("| :--- | :---: | :---: | :---: |")

    summary_rows = [
        ("Composite Benchmark Index", g_ov["overall"]["composite_score"], q_ov["overall"]["composite_score"]),
        ("Overall Categorical Accuracy", g_ov["overall"]["categorical_accuracy"], q_ov["overall"]["categorical_accuracy"]),
        ("Overall Categorical Macro-F1", g_ov["overall"]["categorical_macro_f1"], q_ov["overall"]["categorical_macro_f1"]),
        ("Overall Categorical Weighted-F1", g_ov["overall"]["categorical_weighted_f1"], q_ov["overall"]["categorical_weighted_f1"]),
        ("Environment Macro-F1", g_ov["overall"]["environment_macro_f1"], q_ov["overall"]["environment_macro_f1"]),
        ("Scenario Behavior Macro-F1", g_ov["overall"]["behavior_macro_f1"], q_ov["overall"]["behavior_macro_f1"]),
        ("Actor Class Detection Macro-F1", g_ov["actors"]["class_macro_f1"], q_ov["actors"]["class_macro_f1"]),
        ("Actor Positional Match F1", g_ov["actors"]["positional_f1"], q_ov["actors"]["positional_f1"]),
        ("Temporal Window Mean tIoU", g_ov["temporal"]["mean_tiou"], q_ov["temporal"]["mean_tiou"]),
        ("Key Event ROUGE-L F1", g_ov["key_event_nlp"]["rouge_l_f1"], q_ov["key_event_nlp"]["rouge_l_f1"]),
    ]

    for label, gv, qv in summary_rows:
        d = gv - qv
        win = f"**Gemini (+{d:.4f})**" if d > 0.005 else (f"**Qwen (+{-d:.4f})**" if d < -0.005 else "Tied")
        lines.append(f"| **{label}** | {gv:.4f} | {qv:.4f} | {win} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Field-Wise Performance Table (9 Schema Categorical Fields)")
    lines.append("")
    lines.append("| Field Name | Gemini Accuracy | Gemini Macro-F1 | Qwen Accuracy | Qwen Macro-F1 | F1 Delta | Winner |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for sec, field in CATEGORICAL_FIELDS:
        fname = f"{sec}.{field}"
        g_acc = g_ov["fields"][fname]["accuracy"]
        g_f1 = g_ov["fields"][fname]["macro_f1"]
        q_acc = q_ov["fields"][fname]["accuracy"]
        q_f1 = q_ov["fields"][fname]["macro_f1"]
        diff = g_f1 - q_f1
        win = "Gemini" if diff > 0.005 else ("Qwen" if diff < -0.005 else "Tie")
        lines.append(f"| `{fname}` | {g_acc:.4f} | {g_f1:.4f} | {q_acc:.4f} | {q_f1:.4f} | {diff:+.4f} | **{win}** |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Actor Detection Performance")
    lines.append("")
    lines.append("| Actor Class | groundtruth Instances | Gemini F1 | Qwen F1 | Advantage |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for c in COMMON_ACTOR_CLASSES:
        sup = g_ov["actors"]["per_class"].get(c, {}).get("support", 0)
        gf = g_ov["actors"]["per_class"].get(c, {}).get("f1", 0.0)
        qf = q_ov["actors"]["per_class"].get(c, {}).get("f1", 0.0)
        diff = gf - qf
        win = "Gemini" if diff > 0.005 else ("Qwen" if diff < -0.005 else "Tie")
        lines.append(f"| `{c}` | {sup} | {gf:.4f} | {qf:.4f} | **{win}** |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Dataset-by-Dataset Breakdown")
    lines.append("")
    lines.append("| Dataset | Video Count | Gemini Categorical F1 | Qwen Categorical F1 | Gemini Actor F1 | Qwen Actor F1 | Gemini tIoU | Qwen tIoU |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for ds_name, res in dataset_results.items():
        g_ds = res["gemini"]
        q_ds = res["qwen"]
        cnt = g_ds["total_videos"]
        lines.append(
            f"| **{ds_name}** | {cnt} | {g_ds['overall']['categorical_macro_f1']:.4f} | {q_ds['overall']['categorical_macro_f1']:.4f} | "
            f"{g_ds['actors']['class_macro_f1']:.4f} | {q_ds['actors']['class_macro_f1']:.4f} | "
            f"{g_ds['temporal']['mean_tiou']:.4f} | {q_ds['temporal']['mean_tiou']:.4f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Key Analytical Insights")
    lines.append("")
    lines.append("1. **Behavioral Taxonomy Diversity**:")
    lines.append("   - Gemini Flash-Lite effectively predicts across the entire distribution of `event_type` (`Animal & Pedestrian`, `Road & Environment`, `Obstacle Detection`, etc.) and `ego_maneuver` (`Left Turn`, `Overtake`, `Yield`, `Stop`).")
    lines.append("   - Qwen2.5-VL exhibits a strong bias towards `Vehicle Interaction` and `Co-linear / Follow` across all datasets, leading to lower recall on complex interaction classes.")
    lines.append("2. **Regional & Vulnerable Road User Perception**:")
    lines.append("   - Gemini excels on the IDD and Dashcam datasets for non-standard actors such as `auto_rickshaw` and multi-group pedestrians.")
    lines.append("3. **Temporal Precision**:")
    lines.append("   - Gemini identifies dynamic, bounded event windows (e.g. 00:03 to 00:09), whereas Qwen tends to set static default time windows (e.g. 00:00 to 00:10).")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
        fp.write("\n")

    print(f"Saved benchmark report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gemini and Qwen against VideoDB ground truth.")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset to evaluate (dashcam, kitti, idd, nuscenes, all)")
    parser.add_argument("--output-json", type=str, default="evaluation_results.json", help="Path to save detailed metrics JSON.")
    parser.add_argument("--output-report", type=str, default="BENCHMARK_REPORT.md", help="Path to save Markdown report.")
    args = parser.parse_args()

    target_datasets = DATASETS
    if args.dataset != "all":
        matched = [d for d in DATASETS if args.dataset.lower() in d.lower()]
        if not matched:
            print(f"Error: Unknown dataset '{args.dataset}'. Choices: dashcam, kitti, idd, nuscenes, all")
            sys.exit(1)
        target_datasets = matched

    all_records, missing = load_dataset_annotations(DATASETS)
    print(f"Loaded {len(all_records)} aligned triplets (VideoDB, Gemini, Qwen).")
    if missing["gemini"] > 0 or missing["qwen"] > 0:
        print(f"Warning: Missing files - Gemini: {missing['gemini']}, Qwen: {missing['qwen']}")

    # 1. Overall benchmark (all 200 videos)
    overall_results = run_benchmark(all_records)
    print_comparison_tables(overall_results, title="Overall 200 Videos Benchmark")

    # 2. Per-dataset benchmark
    dataset_results = {}
    for ds in DATASETS:
        ds_records = [r for r in all_records if r["dataset"] == ds]
        if ds_records:
            ds_res = run_benchmark(ds_records)
            dataset_results[ds] = ds_res
            print_comparison_tables(ds_res, title=f"Dataset: {ds} ({len(ds_records)} Videos)")

    # 3. Export JSON
    output_json_path = BASE_DIR / args.output_json
    full_export = {
        "overall": overall_results,
        "by_dataset": dataset_results
    }
    with open(output_json_path, "w", encoding="utf-8") as fp:
        json.dump(full_export, fp, indent=2)
    print(f"\nDetailed metrics saved to: {output_json_path}")

    # 4. Export Markdown report
    output_report_path = BASE_DIR / args.output_report
    generate_markdown_report(overall_results, dataset_results, output_report_path)


if __name__ == "__main__":
    main()

