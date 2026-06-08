"""
Exploratory Data Analysis (EDA) for the HAM10000 dataset.

Usage:
    python src/eda.py --data-dir data --out-dir docs/figs

Outputs (saved to docs/figs/):
    class_distribution.png   -- bar chart with absolute and % counts
    metadata_distributions.png -- age, sex, localization, dx_type
    sample_images.png        -- grid of example images per class
    class_imbalance.png      -- imbalance ratio vs. majority class
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CLASSES_LABEL = {
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "bcc":   "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "vasc":  "Vascular Lesion",
}

PALETTE = sns.color_palette("tab10", len(CLASSES_LABEL))


def load_metadata(data_dir: Path) -> pd.DataFrame:
    meta = next(data_dir.rglob("*metadata*.csv"), None)
    if meta is None:
        raise FileNotFoundError(f"Metadata CSV not found under {data_dir}")
    df = pd.read_csv(meta)
    df["dx_label"] = df["dx"].map(CLASSES_LABEL)
    return df


def plot_class_distribution(df: pd.DataFrame, out_dir: Path):
    counts = df["dx"].value_counts()
    labels = [CLASSES_LABEL[c] for c in counts.index]
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts.values, color=PALETTE)
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("HAM10000 — Class Distribution (n=10,015)", fontsize=13)
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = out_dir / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_metadata_distributions(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Age distribution
    df["age"].dropna().plot.hist(bins=20, ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Age Distribution")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")

    # Sex distribution
    sex_counts = df["sex"].value_counts()
    axes[1].pie(
        sex_counts.values,
        labels=sex_counts.index,
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452", "#55A868"],
        startangle=90,
    )
    axes[1].set_title("Sex Distribution")

    # Localization
    loc = df["localization"].value_counts().head(10)
    loc.plot.barh(ax=axes[2], color="teal")
    axes[2].set_title("Top-10 Localizations")
    axes[2].set_xlabel("Count")
    axes[2].invert_yaxis()

    plt.suptitle("HAM10000 — Metadata Distributions", fontsize=13)
    plt.tight_layout()
    out = out_dir / "metadata_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_imbalance(df: pd.DataFrame, out_dir: Path):
    counts = df["dx"].value_counts()
    majority = counts.max()
    ratios = majority / counts
    labels = [CLASSES_LABEL[c] for c in ratios.index]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, ratios.values, color=PALETTE)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Balanced baseline")
    ax.set_title("Class Imbalance Ratio (majority / class count)")
    ax.set_ylabel("Imbalance ratio")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    out = out_dir / "class_imbalance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_sample_images(data_dir: Path, out_dir: Path, n_per_class: int = 3):
    classes = list(CLASSES_LABEL.keys())
    organized = data_dir / "organized" / "train"
    if not organized.exists():
        print("Skipping sample images — run prepare_dataset.py first.")
        return

    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 3, len(classes) * 3))
    for row, cls in enumerate(classes):
        imgs = list((organized / cls).glob("*.jpg"))[:n_per_class]
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(imgs):
                img = cv2.cvtColor(cv2.imread(str(imgs[col])), cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(CLASSES_LABEL[cls], fontsize=9, rotation=90, labelpad=5)

    plt.suptitle("HAM10000 — Sample Images per Class", fontsize=13)
    plt.tight_layout()
    out = out_dir / "sample_images.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_summary(df: pd.DataFrame):
    print("\n=== HAM10000 EDA Summary ===")
    print(f"Total images : {len(df)}")
    print(f"Unique lesions: {df['lesion_id'].nunique()}")
    print(f"\nClass distribution:")
    counts = df["dx"].value_counts()
    for cls, n in counts.items():
        print(f"  {CLASSES_LABEL[cls]:<25} ({cls}): {n:>5}  ({n/len(df)*100:.1f}%)")
    print(f"\nAge — mean: {df['age'].mean():.1f}  std: {df['age'].std():.1f}  "
          f"missing: {df['age'].isna().sum()}")
    print(f"Sex — {dict(df['sex'].value_counts())}")
    print(f"\nImages with duplicated lesion_id (multi-image lesions): "
          f"{(df['lesion_id'].value_counts() > 1).sum()}")
    majority = counts.max()
    print(f"\nImbalance ratio (max/min): {majority / counts.min():.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="docs/figs")
    parser.add_argument("--n-per-class", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(data_dir)
    print_summary(df)
    plot_class_distribution(df, out_dir)
    plot_metadata_distributions(df, out_dir)
    plot_imbalance(df, out_dir)
    plot_sample_images(data_dir, out_dir, args.n_per_class)
    print(f"\nAll figures saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
