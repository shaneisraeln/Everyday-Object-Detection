"""Automated Dataset Preparation Pipeline for Everyday Desk & Personal Objects.

Downloads hundreds of diverse real-world desk and everyday object images,
applies YOLO-World zero-shot pseudo-labeling with confidence filtering,
incorporates verified local datasets and COCO annotations,
performs data augmentation, and formats into a clean YOLO dataset.

Usage:
    python prepare_dataset.py --target-per-query 8 --augment
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import requests
import yaml
from ultralytics import YOLOWorld


# Target 16 Everyday Object Classes
EVERYDAY_CLASSES = [
    "pen",          # 0
    "pencil",       # 1
    "keys",         # 2
    "wallet",       # 3
    "charger",      # 4
    "earbuds",      # 5
    "calculator",   # 6
    "eyeglasses",   # 7
    "notebook",     # 8
    "mug",          # 9
    "bottle",       # 10
    "phone",        # 11
    "laptop",       # 12
    "scissors",     # 13
    "book",         # 14
    "backpack",     # 15
]

# Prompts for YOLO-World zero-shot extraction
YOLO_WORLD_PROMPTS = [
    "ballpoint pen",
    "pen",
    "pencil",
    "keys",
    "keychain",
    "door key",
    "leather wallet",
    "wallet",
    "purse",
    "phone charger",
    "usb charger",
    "charger cable",
    "wireless earbuds",
    "earphones",
    "headphones",
    "pocket calculator",
    "calculator",
    "eyeglasses",
    "reading glasses",
    "spectacles",
    "sunglasses",
    "spiral notebook",
    "notebook",
    "notepad",
    "coffee mug",
    "ceramic mug",
    "water bottle",
    "bottle",
    "mobile phone",
    "smartphone",
    "laptop computer",
    "laptop",
    "scissors",
    "book",
    "backpack",
]

# Mapping from prompt to target index
PROMPT_TO_CLASS_IDX = {
    "ballpoint pen": 0,
    "pen": 0,
    "pencil": 1,
    "keys": 2,
    "keychain": 2,
    "door key": 2,
    "leather wallet": 3,
    "wallet": 3,
    "purse": 3,
    "phone charger": 4,
    "usb charger": 4,
    "charger cable": 4,
    "wireless earbuds": 5,
    "earphones": 5,
    "headphones": 5,
    "pocket calculator": 6,
    "calculator": 6,
    "eyeglasses": 7,
    "reading glasses": 7,
    "spectacles": 7,
    "sunglasses": 7,
    "spiral notebook": 8,
    "notebook": 8,
    "notepad": 8,
    "coffee mug": 9,
    "ceramic mug": 9,
    "mug": 9,
    "water bottle": 10,
    "bottle": 10,
    "mobile phone": 11,
    "smartphone": 11,
    "phone": 11,
    "laptop computer": 12,
    "laptop": 12,
    "scissors": 13,
    "book": 14,
    "backpack": 15,
}

# Image Search Queries for Wikimedia Commons
SEARCH_QUERIES = {
    "pen": ["ballpoint pen desk", "fountain pen desk", "stationery pens office"],
    "pencil": ["pencils on desk", "pencil notebook table", "colored pencils desk"],
    "keys": ["bunch of keys keychain", "door keys on table", "car keys on desk", "house keys keychain"],
    "wallet": ["leather wallet credit cards", "open pocket wallet table", "men leather wallet", "billfold wallet"],
    "charger": ["phone charger usb cable", "usb wall charger plug", "laptop power adapter charger", "charging cord desk"],
    "earbuds": ["wireless earbuds case", "in-ear earphones desk", "airpods case on table", "bluetooth earphones"],
    "calculator": ["pocket calculator desk", "electronic calculator table", "scientific calculator desk"],
    "eyeglasses": ["reading eyeglasses desk", "spectacles on table", "eyeglasses with case desk"],
    "notebook": ["spiral notebook desk", "open notebook journal pen", "writing pad notebook"],
    "mug": ["coffee mug on desk", "ceramic mug table", "tea mug workspace"],
    "bottle": ["water bottle desk", "reusable drink bottle", "plastic bottle desk"],
    "phone": ["smartphone on desk", "mobile phone table", "cell phone desk"],
    "laptop": ["laptop open desk", "notebook computer workspace", "macbook on desk"],
    "scissors": ["stationery scissors desk", "craft scissors table", "office scissors desk"],
    "book": ["hardcover book desk", "open book table study", "reading book desk"],
    "backpack": ["backpack on chair", "school bag desk", "travel backpack floor"],
    "desk_scenes": [
        "office desk workspace stationery",
        "study desk computer notebook",
        "messy desk stationery supplies",
        "workspace desk gadgets",
        "college study table books stationery",
    ]
}

API_HEADERS = {
    "User-Agent": "EverydayObjectDetector/1.0 (https://github.com/everyday-detection; everyday@detection.org) python-requests/2.32.3"
}


def download_wikimedia_images(query: str, max_count: int, save_dir: Path) -> List[Path]:
    """Download 640px thumbnail images for a search query from Wikimedia Commons."""
    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": min(max_count * 2, 40),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": 640,
    }

    try:
        resp = requests.get(url, params=params, headers=API_HEADERS, timeout=12)
        if resp.status_code != 200:
            return downloaded

        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            if len(downloaded) >= max_count:
                break
            image_info = page_info.get("imageinfo", [{}])[0]
            img_url = image_info.get("thumburl") or image_info.get("url")
            mime = image_info.get("mime", "")

            if not img_url or mime not in ["image/jpeg", "image/png"]:
                continue

            ext = ".jpg" if "jpeg" in mime else ".png"
            img_hash = hashlib.md5(img_url.encode()).hexdigest()[:12]
            target_path = save_dir / f"wiki_{img_hash}{ext}"

            if target_path.exists():
                downloaded.append(target_path)
                continue

            try:
                img_resp = requests.get(img_url, headers=API_HEADERS, timeout=10)
                if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                    with open(target_path, "wb") as f:
                        f.write(img_resp.content)
                    test_img = cv2.imread(str(target_path))
                    if test_img is not None and test_img.shape[0] > 100 and test_img.shape[1] > 100:
                        downloaded.append(target_path)
                    else:
                        target_path.unlink(missing_ok=True)
            except Exception:
                continue

    except Exception as e:
        print(f"  [WARN] Query '{query}' error: {e}", flush=True)

    return downloaded


def augment_sample(
    img: np.ndarray, boxes: List[Tuple[int, float, float, float, float]]
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    """Perform horizontal flip and subtle contrast jitter."""
    flipped_img = cv2.flip(img, 1)
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    aug_img = np.clip(alpha * flipped_img + beta, 0, 255).astype(np.uint8)

    aug_boxes = []
    for cls_idx, x_c, y_c, w, h in boxes:
        new_xc = 1.0 - x_c
        aug_boxes.append((cls_idx, new_xc, y_c, w, h))

    return aug_img, aug_boxes


def build_dataset(
    output_root: Path,
    target_per_query: int = 6,
    confidence_thresh: float = 0.25,
    enable_aug: bool = True,
):
    """Orchestrate dataset download, labeling, augmentation, and split."""
    print("\n" + "=" * 68, flush=True)
    print("  EVERYDAY OBJECTS DATASET INGESTION & AUTO-LABELING PIPELINE", flush=True)
    print("=" * 68, flush=True)
    print(f"  Target Classes ({len(EVERYDAY_CLASSES)}): {', '.join(EVERYDAY_CLASSES)}", flush=True)
    print(f"  Output Directory : {output_root.resolve()}", flush=True)
    print(f"  Target per query : {target_per_query}", flush=True)
    print(f"  Pseudo-label Conf: {confidence_thresh}", flush=True)
    print(f"  Data Augmentation: {enable_aug}", flush=True)
    print("=" * 68 + "\n", flush=True)

    raw_img_dir = output_root / "raw_downloads"
    raw_img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Harvest images from local datasets and web
    print("[1/4] Harvesting images from local and open sources...", flush=True)
    all_image_paths = []

    # Include existing local images: coco128, coco8, sample_images
    existing_dirs = [
        Path("sample_images"),
        Path("datasets/coco128/images/train2017"),
        Path("datasets/coco8/images/train"),
        Path("datasets/coco8/images/val"),
    ]
    for ed in existing_dirs:
        if ed.exists():
            for p in ed.glob("*.*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    all_image_paths.append(p)
    print(f"  Found {len(all_image_paths)} existing local scenes to include.", flush=True)

    category_count = 0
    for category, queries in SEARCH_QUERIES.items():
        category_count += 1
        print(f"  [{category_count}/{len(SEARCH_QUERIES)}] Querying category '{category}'...", flush=True)
        for q in queries:
            downloaded = download_wikimedia_images(q, target_per_query, raw_img_dir)
            all_image_paths.extend(downloaded)
            time.sleep(0.15)

    # Deduplicate image paths
    all_image_paths = list(set(all_image_paths))
    print(f"\n  Total candidate scenes to process: {len(all_image_paths)}", flush=True)

    # 2. Run YOLO-World open-vocabulary annotation
    print("\n[2/4] Running YOLO-World open-vocabulary annotation...", flush=True)
    world_model = YOLOWorld("yolov8s-worldv2.pt")
    world_model.set_classes(YOLO_WORLD_PROMPTS)

    labeled_samples = []

    for idx, img_path in enumerate(all_image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            continue

        try:
            results = world_model.predict(img, conf=confidence_thresh, verbose=False)
            boxes = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for b in results[0].boxes:
                    cls_id = int(b.cls[0].item())
                    prompt_name = YOLO_WORLD_PROMPTS[cls_id]
                    target_idx = PROMPT_TO_CLASS_IDX.get(prompt_name)
                    if target_idx is None:
                        continue

                    xyxy = b.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    xc = max(0.0, min(1.0, xc))
                    yc = max(0.0, min(1.0, yc))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))

                    boxes.append((target_idx, xc, yc, bw, bh))

            if len(boxes) > 0:
                labeled_samples.append((img, boxes))

        except Exception:
            continue

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_image_paths):
            print(f"  Processed {idx + 1}/{len(all_image_paths)} images | Annotated scenes: {len(labeled_samples)}", flush=True)

    print(f"\n  Total high-confidence annotated scenes: {len(labeled_samples)}", flush=True)

    # 3. Augmentation
    if enable_aug and len(labeled_samples) > 0:
        print("\n[3/4] Applying geometric and photometric augmentations...", flush=True)
        augmented_samples = []
        for img, boxes in labeled_samples:
            aug_img, aug_boxes = augment_sample(img, boxes)
            augmented_samples.append((aug_img, aug_boxes))
        labeled_samples.extend(augmented_samples)
        print(f"  Total samples after augmentation: {len(labeled_samples)}", flush=True)
    else:
        print("\n[3/4] Skipping augmentation.", flush=True)

    # 4. Train/Val Split and YOLO dataset structure
    print("\n[4/4] Writing dataset structure and YAML configuration...", flush=True)
    train_img_dir = output_root / "images" / "train"
    val_img_dir = output_root / "images" / "val"
    train_lbl_dir = output_root / "labels" / "train"
    val_lbl_dir = output_root / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(labeled_samples)
    split_idx = int(len(labeled_samples) * 0.85)
    train_samples = labeled_samples[:split_idx]
    val_samples = labeled_samples[split_idx:]

    def save_split(samples, img_dir, lbl_dir, prefix):
        class_counts = {i: 0 for i in range(len(EVERYDAY_CLASSES))}
        for i, (img, boxes) in enumerate(samples):
            stem = f"{prefix}_{i:05d}"
            img_file = img_dir / f"{stem}.jpg"
            lbl_file = lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_file), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(lbl_file, "w") as f:
                for cls_idx, xc, yc, bw, bh in boxes:
                    f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    class_counts[cls_idx] += 1
        return class_counts

    train_counts = save_split(train_samples, train_img_dir, train_lbl_dir, "train")
    val_counts = save_split(val_samples, val_img_dir, val_lbl_dir, "val")

    # Generate dataset.yaml
    dataset_yaml_path = output_root / "dataset.yaml"
    yaml_content = {
        "path": str(output_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(EVERYDAY_CLASSES)},
    }
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print("\n" + "=" * 68, flush=True)
    print("  DATASET PREPARATION COMPLETE", flush=True)
    print("=" * 68, flush=True)
    print(f"  Training Images   : {len(train_samples)}", flush=True)
    print(f"  Validation Images : {len(val_samples)}", flush=True)
    print(f"  Total Images      : {len(labeled_samples)}", flush=True)
    print(f"  Dataset Config    : {dataset_yaml_path.resolve()}", flush=True)
    print("\n  Class Instance Counts (Train):", flush=True)
    for i, name in enumerate(EVERYDAY_CLASSES):
        print(f"    - {name:<12}: {train_counts[i]} instances", flush=True)
    print("=" * 68 + "\n", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Everyday Objects YOLO dataset.")
    parser.add_argument("--output", type=str, default="datasets/everyday_desk", help="Output directory")
    parser.add_argument("--target-per-query", type=int, default=6, help="Target images per search query")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for pseudo-labels")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(
        output_root=Path(args.output),
        target_per_query=args.target_per_query,
        confidence_thresh=args.conf,
        enable_aug=not args.no_augment,
    )
