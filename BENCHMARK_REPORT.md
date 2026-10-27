# Autonomous Driving Video Annotation Benchmark Report

Benchmarking **Google Gemini Flash-Lite** and **Qwen2.5-VL** against **groundtruth** across **200 autonomous driving videos** (Dashcam, KITTI, IDD, NuScenes).

---

## 1. Executive Summary

| Metric Dimension | Gemini Flash-Lite | Qwen2.5-VL | Advantage / Delta |
| :--- | :---: | :---: | :---: |
| **Composite Benchmark Index** | 0.4593 | 0.3541 | **Gemini (+0.1052)** |
| **Overall Categorical Accuracy** | 0.6939 | 0.6194 | **Gemini (+0.0745)** |
| **Overall Categorical Macro-F1** | 0.5004 | 0.3799 | **Gemini (+0.1205)** |
| **Overall Categorical Weighted-F1** | 0.6940 | 0.5320 | **Gemini (+0.1620)** |
| **Environment Macro-F1** | 0.6849 | 0.5943 | **Gemini (+0.0906)** |
| **Scenario Behavior Macro-F1** | 0.2699 | 0.1119 | **Gemini (+0.1580)** |
| **Actor Class Detection Macro-F1** | 0.3760 | 0.1434 | **Gemini (+0.2326)** |
| **Actor Positional Match F1** | 0.4675 | 0.2500 | **Gemini (+0.2175)** |
| **Temporal Window Mean tIoU** | 0.2733 | 0.3155 | **Qwen (+0.0422)** |
| **Key Event ROUGE-L F1** | 0.1779 | 0.1802 | Tied |

---

## 2. Field-Wise Performance Table (9 Schema Categorical Fields)

| Field Name | Gemini Accuracy | Gemini Macro-F1 | Qwen Accuracy | Qwen Macro-F1 | F1 Delta | Winner |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `environment.road_type` | 0.7700 | 0.5844 | 0.8550 | 0.5317 | +0.0527 | **Gemini** |
| `environment.intersection` | 0.8000 | 0.7995 | 0.4650 | 0.3253 | +0.4742 | **Gemini** |
| `environment.traffic_control` | 0.7650 | 0.3748 | 0.7900 | 0.3329 | +0.0419 | **Gemini** |
| `environment.weather` | 0.8550 | 0.8142 | 0.9500 | 0.8945 | -0.0803 | **Qwen** |
| `environment.lighting` | 0.9700 | 0.8514 | 0.9900 | 0.8870 | -0.0356 | **Qwen** |
| `scenario_behavior.event_type` | 0.4900 | 0.3461 | 0.5650 | 0.1203 | +0.2258 | **Gemini** |
| `scenario_behavior.event_pattern` | 0.5700 | 0.2894 | 0.3950 | 0.1416 | +0.1478 | **Gemini** |
| `scenario_behavior.severity` | 0.5000 | 0.2500 | 0.4300 | 0.1503 | +0.0997 | **Gemini** |
| `scenario_behavior.ego_maneuver` | 0.5250 | 0.1942 | 0.1350 | 0.0355 | +0.1587 | **Gemini** |

---

## 3. Actor Detection Performance

| Actor Class | groundtruth Instances | Gemini F1 | Qwen F1 | Advantage |
| :--- | :---: | :---: | :---: | :---: |
| `car` | 180 | 0.9176 | 0.9465 | **Qwen** |
| `motorcycle` | 83 | 0.8590 | 0.5593 | **Gemini** |
| `pedestrian` | 84 | 0.7843 | 0.3366 | **Gemini** |
| `bicycle` | 19 | 0.7097 | 0.1905 | **Gemini** |
| `truck` | 76 | 0.7945 | 0.6222 | **Gemini** |
| `bus` | 36 | 0.7164 | 0.1905 | **Gemini** |
| `auto_rickshaw` | 44 | 0.7708 | 0.0870 | **Gemini** |
| `animal` | 2 | 0.8000 | 0.0000 | **Gemini** |

---

## 4. Dataset-by-Dataset Breakdown

| Dataset | Video Count | Gemini Categorical F1 | Qwen Categorical F1 | Gemini Actor F1 | Qwen Actor F1 | Gemini tIoU | Qwen tIoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **dashcam** | 50 | 0.5308 | 0.4219 | 0.5387 | 0.2967 | 0.2551 | 0.3180 |
| **kitti dataset** | 40 | 0.6326 | 0.4742 | 0.4659 | 0.3862 | 0.3736 | 0.3125 |
| **IDD** | 55 | 0.4884 | 0.3068 | 0.4608 | 0.1718 | 0.2576 | 0.2818 |
| **nuscenes dataset** | 55 | 0.5566 | 0.4678 | 0.3654 | 0.2192 | 0.2326 | 0.3491 |

---

## 5. Key Analytical Insights

1. **Behavioral Taxonomy Diversity**:
   - Gemini Flash-Lite effectively predicts across the entire distribution of `event_type` (`Animal & Pedestrian`, `Road & Environment`, `Obstacle Detection`, etc.) and `ego_maneuver` (`Left Turn`, `Overtake`, `Yield`, `Stop`).
   - Qwen2.5-VL exhibits a strong bias towards `Vehicle Interaction` and `Co-linear / Follow` across all datasets, leading to lower recall on complex interaction classes.
2. **Regional & Vulnerable Road User Perception**:
   - Gemini excels on the IDD and Dashcam datasets for non-standard actors such as `auto_rickshaw` and multi-group pedestrians.
3. **Temporal Precision**:
   - Gemini identifies dynamic, bounded event windows (e.g. 00:03 to 00:09), whereas Qwen tends to set static default time windows (e.g. 00:00 to 00:10).

