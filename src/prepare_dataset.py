"""
Organize HAM10000 images into train/val/test folders by class.

Run after download_data.py:
    python src/prepare_dataset.py

Output structure:
    data/organized/
        train/mel/, train/nv/, ...
        val/mel/, val/nv/, ...
        test/mel/, test/nv/, ...
"""
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "organized"


def find_image(image_id: str) -> Path | None:
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        p = DATA_DIR / part / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def main():
    meta_path = next(DATA_DIR.rglob("*metadata*.csv"), None)
    if meta_path is None:
        raise FileNotFoundError("Metadata CSV not found. Run download_data.py first.")

    df = pd.read_csv(meta_path)
    print(f"Total images in metadata: {len(df)}")
    print(df["dx"].value_counts().to_string())

    # Patient-level split to avoid data leakage between splits
    patients = df.groupby("lesion_id")["dx"].first().reset_index()
    train_p, temp_p = train_test_split(
        patients, test_size=0.30, stratify=patients["dx"], random_state=42
    )
    val_p, test_p = train_test_split(
        temp_p, test_size=0.50, stratify=temp_p["dx"], random_state=42
    )

    split_map = {
        "train": set(train_p["lesion_id"]),
        "val": set(val_p["lesion_id"]),
        "test": set(test_p["lesion_id"]),
    }

    for split in split_map:
        for cls in CLASSES:
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    missing, counts = 0, {s: {c: 0 for c in CLASSES} for s in split_map}

    for _, row in df.iterrows():
        src = find_image(row["image_id"])
        if src is None:
            missing += 1
            continue
        for split, lesion_set in split_map.items():
            if row["lesion_id"] in lesion_set:
                dst = OUT_DIR / split / row["dx"] / f"{row['image_id']}.jpg"
                if not dst.exists():
                    shutil.copy2(src, dst)
                counts[split][row["dx"]] += 1
                break

    if missing:
        print(f"\nWarning: {missing} images not found locally")

    print("\nDataset organized:")
    for split, cls_counts in counts.items():
        total = sum(cls_counts.values())
        print(f"  {split}: {total} images")
        for cls, n in cls_counts.items():
            print(f"    {cls:>6}: {n}")

    print(f"\nOutput: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
