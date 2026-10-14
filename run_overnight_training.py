"""Overnight Training Launcher with Windows Sleep Prevention.

Keeps the laptop awake while training YOLO11m on the everyday objects dataset,
logs milestone metrics, and saves the final best weights to best.pt.

Usage:
    python run_overnight_training.py --epochs 200 --batch 8
"""

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path


# Windows Sleep Prevention Flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def prevent_sleep():
    """Prevent Windows from going to sleep while training overnight."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("  [SYSTEM] Windows sleep prevention enabled (system will stay awake overnight).")
    except Exception as e:
        print(f"  [WARN] Could not set Windows execution state: {e}")


def allow_sleep():
    """Allow Windows to sleep normally once training completes."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("  [SYSTEM] Windows sleep prevention disabled (normal power management restored).")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Overnight YOLO11m Fine-Tuning Launcher.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8 for 6GB VRAM)")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base model checkpoint")
    parser.add_argument("--data", type=str, default="datasets/everyday_desk/dataset.yaml", help="Dataset YAML path")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EVERYDAY OBJECT DETECTION: OVERNIGHT FINE-TUNING RUNNER")
    print("=" * 70)
    print(f"  Backbone Model    : {args.model}")
    print(f"  Target Dataset    : {args.data}")
    print(f"  Total Epochs      : {args.epochs}")
    print(f"  Batch Size        : {args.batch}")
    print("  Estimated Runtime : ~40 to 60 minutes for 200 epochs on RTX 4050")
    print("=" * 70 + "\n")

    prevent_sleep()

    cmd = (
        f'python train.py --data "{args.data}" --model "{args.model}" '
        f'--epochs {args.epochs} --batch {args.batch} --device 0 '
        f'--report-interval 10 --name everyday_objects_fine_tune'
    )
    print(f"  Executing: {cmd}\n")

    start_time = time.time()
    try:
        ret = os.system(cmd)
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        if ret == 0:
            print("  [SUCCESS] Overnight Training Finished Successfully!")
            print(f"  Total Duration: {elapsed / 3600:.2f} hours ({elapsed / 60:.1f} minutes)")
            print("  Fine-tuned model weights saved to: best.pt")
            print("  Run 'python app.py' to test the newly trained detector!")
        else:
            print(f"  [ERROR] Training process exited with code: {ret}")
        print("=" * 70 + "\n")
    finally:
        allow_sleep()


if __name__ == "__main__":
    main()
