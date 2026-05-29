"""
PyTorch Dataset for HAM10000 with augmentation and class-imbalance utilities.
"""
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(split: str, img_size: int = 224) -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class HAM10000Dataset(Dataset):
    def __init__(self, root: str | Path, split: str, img_size: int = 224, transform=None):
        self.transform = transform or get_transforms(split, img_size)
        self.samples: list[tuple[Path, int]] = []

        split_dir = Path(root) / split
        for cls in CLASSES:
            for p in (split_dir / cls).glob("*.jpg"):
                self.samples.append((p, CLASS_TO_IDX[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image=img)["image"]
        return tensor, label


def get_class_weights(dataset: HAM10000Dataset) -> torch.Tensor:
    labels = torch.tensor([label for _, label in dataset.samples])
    counts = torch.bincount(labels, minlength=len(CLASSES)).float()
    return counts.sum() / (len(CLASSES) * counts)


def get_weighted_sampler(dataset: HAM10000Dataset) -> WeightedRandomSampler:
    labels = torch.tensor([label for _, label in dataset.samples])
    counts = torch.bincount(labels, minlength=len(CLASSES)).float()
    weights = 1.0 / counts
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    root = Path(data_dir)
    train_ds = HAM10000Dataset(root, "train", img_size)
    val_ds = HAM10000Dataset(root, "val", img_size)
    test_ds = HAM10000Dataset(root, "test", img_size)

    sampler = get_weighted_sampler(train_ds)
    kw = dict(num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, get_class_weights(train_ds)
