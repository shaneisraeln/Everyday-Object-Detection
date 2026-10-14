"""Turnkey Fine-Tuning Pipeline for YOLO11 on Everyday Objects.

Supports detailed progress tracking, per-epoch progress indicators, and milestone reports every 10 epochs.

Usage:
    python train.py --data coco128.yaml --epochs 50 --model yolo11s.pt --batch 16
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO


class MilestoneReportCallback:
    """Ultralytics training callback that logs clean progress and milestone reports."""

    def __init__(self, total_epochs: int, report_interval: int = 10):
        self.total_epochs = total_epochs
        self.report_interval = report_interval
        self.start_time = time.time()
        self.epoch_start_time = time.time()

    def on_train_epoch_start(self, trainer):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        elapsed = time.time() - self.start_time
        epoch_duration = time.time() - self.epoch_start_time

        # Calculate progress percentage and bar
        pct = (epoch / self.total_epochs) * 100
        bar_len = 25
        filled = int(bar_len * epoch / self.total_epochs)
        bar = "=" * filled + (">" if filled < bar_len else "") + "-" * (bar_len - filled - (1 if filled < bar_len else 0))

        # Extract current losses safely
        loss_items = trainer.loss_items
        if isinstance(loss_items, dict):
            box_loss = float(loss_items.get('box_loss', 0.0))
            cls_loss = float(loss_items.get('cls_loss', 0.0))
            dfl_loss = float(loss_items.get('dfl_loss', 0.0))
        elif hasattr(loss_items, '__len__') and len(loss_items) >= 3:
            box_loss = float(loss_items[0])
            cls_loss = float(loss_items[1])
            dfl_loss = float(loss_items[2])
        else:
            box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0

        print(
            f"  [Epoch {epoch:02d}/{self.total_epochs:02d}] [{bar}] {pct:5.1f}% | "
            f"Box: {box_loss:.4f} | Cls: {cls_loss:.4f} | DFL: {dfl_loss:.4f} | "
            f"Time: {epoch_duration:.1f}s (Total: {elapsed/60:.1f}m)"
        )

        # Print Milestone Report every N epochs or on the final epoch
        if epoch % self.report_interval == 0 or epoch == self.total_epochs:
            self._print_milestone_report(trainer, epoch, elapsed)

    def _print_milestone_report(self, trainer, epoch: int, total_elapsed: float):
        metrics = getattr(trainer, "metrics", {}) or {}
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)

        loss_items = trainer.loss_items
        if isinstance(loss_items, dict):
            box_loss = float(loss_items.get('box_loss', 0.0))
            cls_loss = float(loss_items.get('cls_loss', 0.0))
            dfl_loss = float(loss_items.get('dfl_loss', 0.0))
        elif hasattr(loss_items, '__len__') and len(loss_items) >= 3:
            box_loss = float(loss_items[0])
            cls_loss = float(loss_items[1])
            dfl_loss = float(loss_items[2])
        else:
            box_loss, cls_loss, dfl_loss = 0.0, 0.0, 0.0

        lr = trainer.optimizer.param_groups[0]["lr"] if trainer.optimizer else 0.0

        print("\n" + "=" * 68)
        print(f"  MILESTONE REPORT: EPOCH {epoch} OF {self.total_epochs}")
        print("=" * 68)
        print(f"  Overall Progress    : {epoch / self.total_epochs * 100:.1f}% complete")
        print(f"  Elapsed Time        : {total_elapsed / 60:.2f} minutes")
        print(f"  Learning Rate (LR)  : {lr:.6f}")
        print("  Losses:")
        print(f"    - Bounding Box Loss : {box_loss:.4f}")
        print(f"    - Classification Loss: {cls_loss:.4f}")
        print(f"    - DFL Loss          : {dfl_loss:.4f}")
        print("  Validation Performance:")
        print(f"    - Precision (B)     : {precision:.4f} ({precision:.1%})")
        print(f"    - Recall (B)        : {recall:.4f} ({recall:.1%})")
        print(f"    - mAP@50            : {map50:.4f} ({map50:.1%})")
        print(f"    - mAP@50-95         : {map50_95:.4f} ({map50_95:.1%})")
        print("=" * 68 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11 model on everyday objects dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="coco128.yaml",
        help="Path or name of dataset YAML configuration (default: coco128.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Base pretrained YOLO11 weights (yolo11n.pt, yolo11s.pt, or yolo11m.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: '0' for CUDA GPU, 'cpu' for CPU. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=10,
        help="Epoch frequency to output detailed milestone report (default: 10).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Output project directory for training logs and checkpoints.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="everyday_objects_fine_tune",
        help="Experiment name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 68)
    print("  YOLO11 Everyday Object Detection Fine-Tuning Pipeline")
    print("=" * 68)

    # 1. Device detection
    if not args.device:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"  Target Compute Device : {device}")
    if device == "0" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  CUDA GPU Hardware     : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("  Running on CPU.")

    print(f"  Dataset               : {args.data}")
    print(f"  Base Pretrained Model : {args.model}")
    print(f"  Total Epochs Planned  : {args.epochs}")
    print(f"  Batch Size            : {args.batch}")
    print(f"  Image Resolution      : {args.imgsz}x{args.imgsz}")
    print(f"  Milestone Report Every: {args.report_interval} epochs")
    print("=" * 68 + "\n")

    # 2. Model initialization
    model = YOLO(args.model)

    # 3. Register callback for clean epoch progress and milestone reports
    milestone_cb = MilestoneReportCallback(
        total_epochs=args.epochs,
        report_interval=args.report_interval,
    )
    model.add_callback("on_train_epoch_start", milestone_cb.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", milestone_cb.on_train_epoch_end)

    # 4. Train model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        save=True,
        verbose=False,
    )

    # 5. Export best weights
    best_weights = None
    if hasattr(results, "save_dir") and results.save_dir:
        candidate = Path(results.save_dir) / "weights" / "best.pt"
        if candidate.exists():
            best_weights = candidate

    if not best_weights:
        # Check standard user argument path
        candidate = Path(args.project) / args.name / "weights" / "best.pt"
        if candidate.exists():
            best_weights = candidate

    if not best_weights:
        # Search runs directory for any recently created best.pt
        found = list(Path("runs").glob(f"**/{args.name}/weights/best.pt"))
        if found:
            best_weights = found[0]

    if best_weights and best_weights.exists():
        destination = Path("best.pt").resolve()
        shutil.copy2(best_weights, destination)
        backup_dir = Path("weights")
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(best_weights, backup_dir / f"everyday_{args.name}_best.pt")
        print("\n" + "=" * 68)
        print("  [SUCCESS] Fine-Tuning Complete!")
        print(f"  Best fine-tuned model saved to: {destination}")
        print(f"  Backup model archived to: {backup_dir / f'everyday_{args.name}_best.pt'}")
        print("=" * 68)
        print("\n  The model is now fine-tuned and ready.")
        print("  To activate it in the app, update .env with:")
        print("    YOLO_STANDARD_WEIGHTS=best.pt\n")
    else:
        print("\n[WARNING] Training completed, but weights/best.pt could not be located.")


if __name__ == "__main__":
    main()
