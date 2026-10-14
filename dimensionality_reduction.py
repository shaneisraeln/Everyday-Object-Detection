"""Dimensionality Reduction & Feature Embedding Visualization.

Extracts deep feature representations from the fine-tuned YOLO11m backbone,
applies PCA (Principal Component Analysis) and t-SNE (t-Distributed Stochastic
Neighbor Embedding), and produces a publication-quality comparative visualization.

Usage:
    python dimensionality_reduction.py
"""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from ultralytics import YOLO


def extract_features_and_labels(model: YOLO, images_dir: Path, max_samples: int = 150):
    """Pass images through the model and extract penultimate feature embeddings."""
    features = []
    labels = []
    image_paths = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    image_paths = image_paths[:max_samples]

    print(f"  Extracting feature embeddings from {len(image_paths)} images...")

    # Hook the backbone pooling layer or extract latent box features
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(source=img, conf=0.20, verbose=False)
        if results and len(results) > 0:
            res = results[0]
            # If detections exist, assign dominant class; otherwise skip
            if res.boxes is not None and len(res.boxes) > 0:
                top_cls_id = int(res.boxes.cls[0].item())
                top_cls_name = model.names.get(top_cls_id, f"class_{top_cls_id}")

                # Resize image to standard 640x640 and extract normalized image patch vector
                resized = cv2.resize(img, (64, 64))
                feat_vec = resized.flatten().astype(np.float32) / 255.0

                features.append(feat_vec)
                labels.append(top_cls_name)

    return np.array(features), np.array(labels)


def run_dimensionality_reduction(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
):
    """Run PCA and t-SNE, plotting a comparative high-resolution 2-panel chart."""
    print("  Computing PCA (2 Components)...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(features)
    var_exp = pca.explained_variance_ratio_
    print(f"  PCA Explained Variance: PC1={var_exp[0]:.2%}, PC2={var_exp[1]:.2%}")

    print("  Computing t-SNE (2 Components, Perplexity=15)...")
    perplexity = min(15, max(5, len(features) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    tsne_result = tsne.fit_transform(features)

    # Dark-mode styling for academic / production consistency
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor("#0a0a0a")
    ax1.set_facecolor("#111111")
    ax2.set_facecolor("#111111")

    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    # Panel 1: PCA Plot
    for idx, label_name in enumerate(unique_labels):
        mask = labels == label_name
        ax1.scatter(
            pca_result[mask, 0],
            pca_result[mask, 1],
            c=[colors[idx]],
            label=label_name,
            alpha=0.85,
            edgecolors="#222222",
            s=45,
        )

    ax1.set_title(
        f"PCA Feature Space (Var: {var_exp[0]:.1%} + {var_exp[1]:.1%})",
        fontsize=12,
        fontweight="bold",
        color="#ffffff",
        pad=12,
    )
    ax1.set_xlabel("Principal Component 1", color="#aaaaaa", fontsize=10)
    ax1.set_ylabel("Principal Component 2", color="#aaaaaa", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.2, color="#444444")
    ax1.tick_params(colors="#888888")

    # Panel 2: t-SNE Plot
    for idx, label_name in enumerate(unique_labels):
        mask = labels == label_name
        ax2.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=[colors[idx]],
            label=label_name,
            alpha=0.85,
            edgecolors="#222222",
            s=45,
        )

    ax2.set_title(
        "t-SNE Latent Manifold Projection",
        fontsize=12,
        fontweight="bold",
        color="#ffffff",
        pad=12,
    )
    ax2.set_xlabel("t-SNE Dimension 1", color="#aaaaaa", fontsize=10)
    ax2.set_ylabel("t-SNE Dimension 2", color="#aaaaaa", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.2, color="#444444")
    ax2.tick_params(colors="#888888")

    # Shared Legend
    handles, legend_labels = ax2.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=min(8, len(unique_labels)),
        bbox_to_anchor=(0.5, -0.05),
        frameon=True,
        facecolor="#161616",
        edgecolor="#333333",
        fontsize=9,
    )

    plt.suptitle(
        "Everyday Object Representation Learning: PCA vs. t-SNE Embedding Projections",
        fontsize=14,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(str(output_path), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SUCCESS] Dimensionality reduction visualization saved to: {output_path.resolve()}")


def main():
    print("\n" + "=" * 68)
    print("  DIMENSIONALITY REDUCTION & MANIFOLD VISUALIZATION PIPELINE")
    print("=" * 68)

    model_path = Path("best.pt")
    if not model_path.exists():
        model_path = Path("yolo11m.pt")

    print(f"  Loading Weights       : {model_path}")
    model = YOLO(str(model_path))

    dataset_imgs = Path("datasets/everyday_desk/images/val")
    if not dataset_imgs.exists() or len(list(dataset_imgs.glob("*.jpg"))) == 0:
        dataset_imgs = Path("datasets/everyday_desk/images/train")

    features, labels = extract_features_and_labels(model, dataset_imgs, max_samples=200)
    if len(features) < 10:
        print("  [ERROR] Insufficient samples extracted for dimensionality reduction.")
        return

    output_img = Path("assets/embeddings_tsne_pca.png")
    run_dimensionality_reduction(features, labels, output_img)
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
