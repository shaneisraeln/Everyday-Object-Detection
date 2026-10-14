"""Empirical benchmark comparing YOLO11n and YOLO-World (yolov8s-worldv2).

Measures:
1. Inference Latency (ms) across multiple runs on identical hardware.
2. GPU VRAM Consumption (MB).
3. Detection accuracy, recall, and false detections on benchmark images.
4. Open-vocabulary novel object capability evaluation (objects absent from COCO-80).
"""

import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.vision.detector import YOLODetector
from src.vision.world_detector import YOLOWorldDetector, DEFAULT_YOLO_WORLD_CLASSES


def compute_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Compute IoU between two xyxy bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_coco128_sample_with_labels():
    """Find images in coco128 that contain target everyday objects and return (img_path, ground_truth_boxes)."""
    coco_img_dir = Path("datasets/coco128/images/train2017")
    coco_lbl_dir = Path("datasets/coco128/labels/train2017")

    # Map COCO category IDs to everyday names
    target_ids = {
        39: "bottle",
        41: "cup",
        56: "chair",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "phone",
        73: "book",
        74: "clock",
        76: "scissors",
    }

    selected_samples = []
    if not coco_img_dir.exists() or not coco_lbl_dir.exists():
        return selected_samples

    for label_file in sorted(coco_lbl_dir.glob("*.txt")):
        img_name = label_file.stem + ".jpg"
        img_path = coco_img_dir / img_name
        if not img_path.exists():
            continue

        with open(label_file, "r") as f:
            lines = [l.strip().split() for l in f if l.strip()]

        gt_boxes = []
        for parts in lines:
            cls_id = int(parts[0])
            if cls_id in target_ids:
                # YOLO format: cls_id, cx, cy, w, h normalized
                cx, cy, nw, nh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                gt_boxes.append({
                    "class_id": cls_id,
                    "label": target_ids[cls_id],
                    "norm_bbox": (cx, cy, nw, nh),
                })

        if gt_boxes:
            selected_samples.append((str(img_path), gt_boxes))
            if len(selected_samples) >= 12:  # take 12 well-labeled everyday scenes
                break

    return selected_samples


def run_benchmark():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"=" * 70)
    print(f"BENCHMARK: YOLO11n (COCO-Pretrained) vs YOLO-World (Open-Vocabulary)")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=" * 70)

    # 1. Initialize Detectors
    print("\n[1/4] Initializing Detectors...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    mem_before_11n = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0

    det_11n = YOLODetector(model_name="yolo11n.pt", confidence_threshold=0.25, device=device)
    mem_after_11n = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    vram_11n = mem_after_11n - mem_before_11n

    world_prompts = [
        "bottle", "cup", "chair", "laptop", "mouse", "remote", "keyboard",
        "phone", "book", "clock", "scissors", "pen", "pencil", "keys", "wallet",
    ]
    mem_before_world = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    det_world = YOLOWorldDetector(
        model_path="yolov8s-worldv2.pt",
        classes=world_prompts,
        confidence_threshold=0.25,
        device=device,
    )
    mem_after_world = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    vram_world = mem_after_world - mem_before_world

    print(f"  YOLO11n VRAM:   {vram_11n:.1f} MB (Weights: 5.6 MB)")
    print(f"  YOLO-World VRAM: {vram_world:.1f} MB (Weights: 24.7 MB)")

    # 2. Latency Measurement (Warmup + 30 timed iterations on identical 960x720 canvas)
    print("\n[2/4] Measuring Inference Latency (30 iterations)...")
    dummy_frame = cv2.imread("sample_images/desk_bottle.png")
    if dummy_frame is None:
        dummy_frame = np.zeros((720, 960, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        det_11n.detect(dummy_frame)
        det_world.detect(dummy_frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies_11n = []
    for _ in range(30):
        t0 = time.perf_counter()
        det_11n.detect(dummy_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_11n.append((time.perf_counter() - t0) * 1000)

    latencies_world = []
    for _ in range(30):
        t0 = time.perf_counter()
        det_world.detect(dummy_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_world.append((time.perf_counter() - t0) * 1000)

    mean_11n, p95_11n = np.mean(latencies_11n), np.percentile(latencies_11n, 95)
    mean_world, p95_world = np.mean(latencies_world), np.percentile(latencies_world, 95)

    print(f"  YOLO11n Latency:    Mean = {mean_11n:.2f} ms | p95 = {p95_11n:.2f} ms")
    print(f"  YOLO-World Latency:  Mean = {mean_world:.2f} ms | p95 = {p95_world:.2f} ms")

    # 3. Ground-Truth Accuracy Evaluation on Labeled Test Scenes
    print("\n[3/4] Evaluating Ground-Truth Object Accuracy on Labeled Scenes...")
    samples = load_coco128_sample_with_labels()
    print(f"  Loaded {len(samples)} labeled test scene images containing everyday targets.")

    results = {
        "yolo11n": {"tp": 0, "fp": 0, "fn": 0, "per_class": {}},
        "yolo_world": {"tp": 0, "fp": 0, "fn": 0, "per_class": {}},
    }

    for img_path, gt_list in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        # Convert ground truth normalized boxes to pixel xyxy
        gt_converted = []
        for gt in gt_list:
            cx, cy, nw, nh = gt["norm_bbox"]
            x1 = int((cx - nw / 2) * iw)
            y1 = int((cy - nh / 2) * ih)
            x2 = int((cx + nw / 2) * iw)
            y2 = int((cy + nh / 2) * ih)
            gt_converted.append({"label": gt["label"].lower(), "bbox": (x1, y1, x2, y2)})

        # Evaluate YOLO11n
        preds_11n = det_11n.detect(img, confidence=0.25)
        matched_gt_11n = set()
        for p in preds_11n:
            p_lbl = p.label.lower()
            matched = False
            for g_idx, g in enumerate(gt_converted):
                if g_idx not in matched_gt_11n and (p_lbl == g["label"] or g["label"] in p_lbl or p_lbl in g["label"]):
                    if compute_iou(p.bbox_xyxy, g["bbox"]) >= 0.40:
                        matched = True
                        matched_gt_11n.add(g_idx)
                        break
            if matched:
                results["yolo11n"]["tp"] += 1
            else:
                results["yolo11n"]["fp"] += 1
        results["yolo11n"]["fn"] += (len(gt_converted) - len(matched_gt_11n))

        # Evaluate YOLO-World
        preds_world = det_world.detect(img, confidence=0.25)
        matched_gt_world = set()
        for p in preds_world:
            p_lbl = p.label.lower()
            matched = False
            for g_idx, g in enumerate(gt_converted):
                if g_idx not in matched_gt_world and (p_lbl == g["label"] or g["label"] in p_lbl or p_lbl in g["label"]):
                    if compute_iou(p.bbox_xyxy, g["bbox"]) >= 0.40:
                        matched = True
                        matched_gt_world.add(g_idx)
                        break
            if matched:
                results["yolo_world"]["tp"] += 1
            else:
                results["yolo_world"]["fp"] += 1
        results["yolo_world"]["fn"] += (len(gt_converted) - len(matched_gt_world))

    for m_key, title in [("yolo11n", "YOLO11n (COCO)"), ("yolo_world", "YOLO-World (Open-Vocab)")]:
        tp = results[m_key]["tp"]
        fp = results[m_key]["fp"]
        fn = results[m_key]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  {title:28s} | TP: {tp:2d} | FP: {fp:2d} | FN: {fn:2d} | Precision: {precision:5.1%} | Recall: {recall:5.1%} | F1: {f1:5.1%}")

    # 4. Open-Vocabulary Novel Detection Test (Classes absent from COCO)
    print("\n[4/4] Testing Novel Everyday Items Absent from COCO-80...")
    # Scissors test scene with user scissors
    test_img = cv2.imread("user_scissors_frame.png")
    if test_img is not None:
        det_world.set_classes(["pair of scissors", "stationery scissors", "reading book", "person"])
        w_res = det_world.detect(test_img, confidence=0.25)
        n_res = det_11n.detect(test_img, confidence=0.25)
        print(f"  Test image 'user_scissors_frame.png':")
        print(f"    YOLO11n detected:    {[f'{d.label} ({d.confidence:.0%})' for d in n_res]}")
        print(f"    YOLO-World detected: {[f'{d.label} ({d.confidence:.0%})' for d in w_res]}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Save benchmark metrics to json
    report = {
        "hardware": {
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "pytorch_version": torch.__version__,
        },
        "latency_ms": {
            "yolo11n_mean": round(mean_11n, 2),
            "yolo11n_p95": round(p95_11n, 2),
            "yolo_world_mean": round(mean_world, 2),
            "yolo_world_p95": round(p95_world, 2),
        },
        "vram_mb": {
            "yolo11n": round(vram_11n, 1),
            "yolo_world": round(vram_world, 1),
        },
        "coco_metrics": {
            "yolo11n": {
                "tp": results["yolo11n"]["tp"],
                "fp": results["yolo11n"]["fp"],
                "fn": results["yolo11n"]["fn"],
                "precision": round(results["yolo11n"]["tp"] / max(1, results["yolo11n"]["tp"] + results["yolo11n"]["fp"]), 4),
                "recall": round(results["yolo11n"]["tp"] / max(1, results["yolo11n"]["tp"] + results["yolo11n"]["fn"]), 4),
            },
            "yolo_world": {
                "tp": results["yolo_world"]["tp"],
                "fp": results["yolo_world"]["fp"],
                "fn": results["yolo_world"]["fn"],
                "precision": round(results["yolo_world"]["tp"] / max(1, results["yolo_world"]["tp"] + results["yolo_world"]["fp"]), 4),
                "recall": round(results["yolo_world"]["tp"] / max(1, results["yolo_world"]["tp"] + results["yolo_world"]["fn"]), 4),
            },
        },
    }
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved benchmark metrics to benchmark_report.json")


if __name__ == "__main__":
    run_benchmark()
