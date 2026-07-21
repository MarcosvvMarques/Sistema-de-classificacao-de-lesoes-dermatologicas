"""Indexação, divisão e carregamento do HAM10000."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CLASS_NAMES, CLASS_TO_IDX

REQUIRED_COLUMNS = {
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
}


def find_metadata(data_dir: Path) -> Path:
    candidates = sorted(data_dir.rglob("*metadata*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Metadados não encontrados em {data_dir}. Execute download_data.py."
        )
    return candidates[0]


def build_image_index(data_dir: Path) -> dict[str, Path]:
    paths = [
        path
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for path in data_dir.rglob(pattern)
    ]
    index: dict[str, Path] = {}
    for path in paths:
        image_id = path.stem
        if image_id in index and index[image_id] != path:
            raise ValueError(f"image_id duplicado: {image_id}")
        index[image_id] = path.resolve()
    return index


def load_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = find_metadata(data_dir)
    frame = pd.read_csv(metadata_path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes: {sorted(missing_columns)}")

    unknown_classes = set(frame["dx"].dropna().unique()).difference(CLASS_NAMES)
    if unknown_classes:
        raise ValueError(f"Classes desconhecidas: {sorted(unknown_classes)}")

    image_index = build_image_index(data_dir)
    frame = frame.copy()
    frame["image_path"] = frame["image_id"].map(image_index)
    missing_images = frame.loc[frame["image_path"].isna(), "image_id"].tolist()
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} imagens dos metadados não foram encontradas: {preview}"
        )
    frame["label"] = frame["dx"].map(CLASS_TO_IDX).astype("int64")
    return frame


def create_grouped_splits(
    frame: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> pd.DataFrame:
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("As frações devem ser positivas e somar menos de 1.")

    labels_per_lesion = frame.groupby("lesion_id")["dx"].nunique()
    if (labels_per_lesion > 1).any():
        raise ValueError("Uma mesma lesion_id possui diagnósticos conflitantes.")

    lesions = frame[["lesion_id", "dx"]].drop_duplicates("lesion_id")
    train_lesions, remaining = train_test_split(
        lesions,
        train_size=train_fraction,
        random_state=seed,
        stratify=lesions["dx"],
    )
    relative_val_size = val_fraction / (1.0 - train_fraction)
    val_lesions, test_lesions = train_test_split(
        remaining,
        train_size=relative_val_size,
        random_state=seed,
        stratify=remaining["dx"],
    )

    split_by_lesion = {
        **dict.fromkeys(train_lesions["lesion_id"], "train"),
        **dict.fromkeys(val_lesions["lesion_id"], "val"),
        **dict.fromkeys(test_lesions["lesion_id"], "test"),
    }
    result = frame.copy()
    result["split"] = result["lesion_id"].map(split_by_lesion)
    validate_splits(result)
    return result


def validate_splits(frame: pd.DataFrame) -> None:
    expected = {"train", "val", "test"}
    found = set(frame["split"].dropna().unique())
    if found != expected:
        raise ValueError(f"Splits inválidos: esperado {expected}, encontrado {found}")
    overlap = frame.groupby("lesion_id")["split"].nunique()
    if (overlap > 1).any():
        raise ValueError("Data leakage: uma lesion_id aparece em múltiplos splits.")
    if frame["split"].isna().any():
        raise ValueError("Há amostras sem split.")


def prepare_splits(data_dir: Path, output_path: Path, seed: int = 42) -> pd.DataFrame:
    frame = create_grouped_splits(load_metadata(data_dir), seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    serializable["image_path"] = serializable["image_path"].astype(str)
    serializable.to_csv(output_path, index=False)
    return frame


def build_transforms(image_size: int, training: bool) -> A.Compose:
    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if training:
        return A.Compose(
            [
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, p=0.4),
                A.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.05,
                    p=0.4,
                ),
                normalize,
                ToTensorV2(),
            ],
            seed=42,
        )
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            normalize,
            ToTensorV2(),
        ]
    )


class HAM10000Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, training: bool = False):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = build_transforms(image_size, training=training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Não foi possível abrir {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=image)["image"]
        return {
            "image": tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
        }


def class_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    if (counts == 0).any():
        absent = [CLASS_NAMES[index] for index, count in counts.items() if count == 0]
        raise ValueError(f"Classes ausentes no treino: {absent}")
    weights = len(frame) / (len(CLASS_NAMES) * counts.to_numpy(dtype=np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    imbalance_strategy: Literal["class_weights", "sampler", "none"] = "class_weights",
) -> dict[str, DataLoader]:
    validate_splits(frame)
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_frame = frame[frame["split"] == split].reset_index(drop=True)
        dataset = HAM10000Dataset(
            split_frame,
            image_size=image_size,
            training=split == "train",
        )
        sampler = None
        shuffle = split == "train"
        if split == "train" and imbalance_strategy == "sampler":
            counts = split_frame["label"].value_counts()
            sample_weights = split_frame["label"].map(lambda value: 1.0 / counts[value])
            sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double),
                num_samples=len(split_frame),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
    return loaders
