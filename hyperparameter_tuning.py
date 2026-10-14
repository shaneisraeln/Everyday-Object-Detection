"""Automated Hyperparameter Optimization Pipeline using Optuna.

Performs Bayesian optimization on key YOLO11 training hyperparameters:
- Initial learning rate (lr0)
- Weight decay
- Momentum
- Bounding box loss gain (box)
- Classification loss gain (cls)

Logs trial telemetry, saves best parameters to optuna_study_results.json,
and generates an optimization history visualization in assets/optuna_study.png.

Usage:
    python hyperparameter_tuning.py --n-trials 5 --epochs 2
"""

import argparse
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import optuna
import torch
from ultralytics import YOLO


def objective(trial: optuna.Trial, data_yaml: str, epochs: int, device: str) -> float:
    """Optuna objective function maximizing validation mAP@50."""
    # Suggest continuous and categorical hyperparameters
    lr0 = trial.suggest_float("lr0", 1e-4, 1e-2, log=True)
    lrf = trial.suggest_float("lrf", 0.01, 0.2)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    momentum = trial.suggest_float("momentum", 0.85, 0.98)
    box_gain = trial.suggest_float("box", 5.0, 10.0)
    cls_gain = trial.suggest_float("cls", 0.3, 1.5)

    trial_name = f"optuna_trial_{trial.number}"
    print(f"\n--- [TRIAL {trial.number}] lr0={lr0:.5f}, wd={weight_decay:.5f}, momentum={momentum:.3f} ---")

    # Load baseline model
    model = YOLO("yolo11s.pt" if Path("yolo11s.pt").exists() else "yolo11n.pt")

    # Run short training trial
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=8,
            imgsz=640,
            device=device,
            project="runs/optuna",
            name=trial_name,
            lr0=lr0,
            lrf=lrf,
            weight_decay=weight_decay,
            momentum=momentum,
            box=box_gain,
            cls=cls_gain,
            verbose=False,
            save=False,
            plots=False,
        )

        # Retrieve validation mAP50 metric
        metrics = getattr(results, "results_dict", {}) or {}
        map50 = float(metrics.get("metrics/mAP50(B)", 0.0))

        # Cleanup trial files to conserve disk space
        trial_dir = Path("runs/optuna") / trial_name
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)

        return map50

    except Exception as e:
        print(f"  [ERROR] Trial {trial.number} failed: {e}")
        return 0.0


def plot_optuna_history(study: optuna.Study, output_path: Path):
    """Plot optimization history and parameter values across trials."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    fig.patch.set_facecolor("#0a0a0a")
    ax1.set_facecolor("#111111")
    ax2.set_facecolor("#111111")

    trial_numbers = [t.number for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    best_values = []
    current_best = 0.0
    for v in values:
        current_best = max(current_best, v)
        best_values.append(current_best)

    # Panel 1: Objective Metric vs Trial
    ax1.plot(trial_numbers, values, "o-", color="#3b82f6", label="Trial mAP@50", markersize=6)
    ax1.plot(trial_numbers, best_values, "--", color="#10b981", label="Best mAP@50 Found", linewidth=2)
    ax1.set_title("Optuna Bayesian Optimization History", color="#ffffff", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Trial Number", color="#aaaaaa")
    ax1.set_ylabel("Validation mAP@50", color="#aaaaaa")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#444444")
    ax1.legend(facecolor="#161616", edgecolor="#333333")

    # Panel 2: Learning Rate vs Objective
    lrs = [t.params.get("lr0", 0.0) for t in study.trials if t.value is not None]
    scatter = ax2.scatter(lrs, values, c=values, cmap="viridis", s=60, edgecolors="#ffffff", alpha=0.9)
    ax2.set_xscale("log")
    ax2.set_title("Learning Rate (lr0) Exploration vs mAP@50", color="#ffffff", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Initial Learning Rate (Log Scale)", color="#aaaaaa")
    ax2.set_ylabel("Validation mAP@50", color="#aaaaaa")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#444444")
    fig.colorbar(scatter, ax=ax2, label="mAP@50")

    plt.suptitle("Hyperparameter Tuning: Bayesian Search over YOLO11 Latent Space", color="#ffffff", fontsize=13, y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(output_path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization.")
    parser.add_argument("--n-trials", type=int, default=5, help="Number of search trials")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per trial")
    parser.add_argument("--data", type=str, default="datasets/everyday_desk/dataset.yaml", help="Dataset YAML")
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  OPTUNA HYPERPARAMETER TUNING PIPELINE (YOLO11)")
    print("=" * 68)
    print(f"  Target Trials : {args.n_trials}")
    print(f"  Epochs / Trial: {args.epochs}")
    print(f"  Dataset Config: {args.data}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"  Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 68 + "\n")

    # Set up Optuna study
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="yolo11_everyday_hpo",
    )

    study.optimize(
        lambda trial: objective(trial, args.data, args.epochs, device),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 68)
    print("  OPTIMIZATION STUDY COMPLETE")
    print("=" * 68)
    print(f"  Best Trial Number : {study.best_trial.number}")
    print(f"  Best mAP@50 Value : {study.best_value:.4f}")
    print("  Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    - {k:<15}: {v}")

    # Export results JSON
    results_json = {
        "best_trial_number": study.best_trial.number,
        "best_map50": study.best_value,
        "best_parameters": study.best_params,
        "total_trials": len(study.trials),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "duration_seconds": t.duration.total_seconds() if t.duration else 0.0,
            }
            for t in study.trials
        ],
    }
    with open("optuna_study_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("\n  Saved trial telemetry to: optuna_study_results.json")

    # Plot history
    chart_path = Path("assets/optuna_study.png")
    plot_optuna_history(study, chart_path)
    print(f"  Saved optimization plot to: {chart_path.resolve()}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
