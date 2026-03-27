#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAINABLE_MODELS = ("gil2023", "dm_qpmnet", "pg_diffseg")
NON_TRAINABLE_MODELS = ("cellsnap2024", "chen2024")
ALL_MODELS = TRAINABLE_MODELS + NON_TRAINABLE_MODELS


@dataclass
class SampleRecord:
    group_id: str
    sample_id: str
    primary_path: str
    mask_paths: List[str]
    secondary_path: Optional[str] = None


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_miou: float
    val_dice: float
    val_precision: float
    val_recall: float
    val_f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QPI paper-inspired models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TRAINABLE_MODELS),
        choices=ALL_MODELS,
        help="Model ids to run. Default: all trainable models.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument("--output-root", type=Path, default=Path("./runs/paper_models"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--num-val-groups", type=int, default=2)
    parser.add_argument(
        "--val-groups",
        nargs="*",
        default=None,
        help="Optional explicit validation group ids. If omitted, exactly 2 groups are chosen automatically.",
    )
    parser.add_argument(
        "--min-auto-val-group-size",
        type=int,
        default=5,
        help="Groups smaller than this are excluded from automatic validation-group sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--dm-secondary-mode",
        choices=("duplicate", "required"),
        default="duplicate",
        help="How DM-QPMNet gets a second modality when only one modality exists locally.",
    )
    parser.add_argument(
        "--secondary-image-pattern",
        type=str,
        default=None,
        help="Optional glob pattern inside each sample's images folder for a true second modality.",
    )
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--diffusion-sample-steps", type=int, default=25)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_model_defaults(model_name: str, args: argparse.Namespace) -> Dict[str, float | int]:
    defaults = {
        "gil2023": {"epochs": 200, "batch_size": 5, "lr": 1e-2, "optimizer": "sgd"},
        "dm_qpmnet": {"epochs": 80, "batch_size": 4, "lr": 1e-3, "optimizer": "adamw"},
        "pg_diffseg": {"epochs": 60, "batch_size": 4, "lr": 2e-4, "optimizer": "adamw"},
    }
    cfg = defaults[model_name].copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    return cfg


def normalize_group_id_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)


def collect_samples(
    data_root: Path,
    secondary_image_pattern: Optional[str],
    dm_secondary_mode: str,
) -> List[SampleRecord]:
    samples: List[SampleRecord] = []
    for sample_dir in sorted(data_root.glob("*/*")):
        if not sample_dir.is_dir():
            continue
        image_candidates = sorted((sample_dir / "images").glob("*_mSLIM.png"))
        mask_candidates = sorted((sample_dir / "masks").glob("*.png"))
        if not image_candidates or not mask_candidates:
            continue

        secondary_path = None
        if secondary_image_pattern:
            secondary_candidates = sorted((sample_dir / "images").glob(secondary_image_pattern))
            if secondary_candidates:
                secondary_path = str(secondary_candidates[0])
        elif dm_secondary_mode == "duplicate":
            secondary_path = str(image_candidates[0])

        samples.append(
            SampleRecord(
                group_id=sample_dir.parent.name.zfill(3),
                sample_id=sample_dir.name,
                primary_path=str(image_candidates[0]),
                secondary_path=secondary_path,
                mask_paths=[str(x) for x in mask_candidates],
            )
        )
    if not samples:
        raise FileNotFoundError(f"No samples found under {data_root}")
    return samples


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1, 99))
    if high > low:
        image = np.clip((image - low) / (high - low), 0.0, 1.0)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image


def load_and_resize_image(path: str, image_size: int) -> np.ndarray:
    image = np.array(Image.open(path), dtype=np.float32)
    image = resize(
        image,
        (image_size, image_size),
        order=1,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    return normalize_image(image)


def load_and_resize_mask(mask_paths: Sequence[str], image_size: int) -> np.ndarray:
    merged = np.zeros((image_size, image_size), dtype=np.uint8)
    for mask_path in mask_paths:
        mask_arr = np.array(Image.open(mask_path), dtype=np.uint8)
        mask_arr = resize(
            mask_arr,
            (image_size, image_size),
            order=0,
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)
        class_value = 2 if "Prot" in Path(mask_path).name else 1
        merged = np.maximum(merged, np.where(mask_arr > 0, class_value, 0).astype(np.uint8))
    return merged


def preprocess_samples(
    samples: Sequence[SampleRecord],
    image_size: int,
    require_secondary: bool,
    cache_dir: Optional[Path] = None,
    rebuild_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        primary_path = cache_dir / f"primary_{image_size}.npy"
        secondary_path = cache_dir / f"secondary_{image_size}.npy"
        masks_path = cache_dir / f"masks_{image_size}.npy"
        manifest_path = cache_dir / f"manifest_{image_size}.csv"
        if (
            not rebuild_cache
            and primary_path.exists()
            and secondary_path.exists()
            and masks_path.exists()
            and manifest_path.exists()
        ):
            manifest = pd.read_csv(manifest_path)
            manifest["group_id"] = normalize_group_id_series(manifest["group_id"])
            return (
                np.load(primary_path),
                np.load(secondary_path),
                np.load(masks_path),
                manifest,
            )

    primary_images: List[np.ndarray] = []
    secondary_images: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    rows: List[Dict[str, str]] = []

    for record in samples:
        primary = load_and_resize_image(record.primary_path, image_size)
        secondary = None
        if record.secondary_path:
            secondary = load_and_resize_image(record.secondary_path, image_size)
        elif require_secondary:
            raise ValueError(
                f"Sample {record.sample_id} has no secondary modality. "
                "Provide --secondary-image-pattern or use --dm-secondary-mode duplicate."
            )

        mask = load_and_resize_mask(record.mask_paths, image_size)
        primary_images.append(primary)
        secondary_images.append(secondary if secondary is not None else primary.copy())
        masks.append(mask)
        rows.append(
            {
                "group_id": record.group_id,
                "sample_id": record.sample_id,
                "primary_path": record.primary_path,
                "secondary_path": record.secondary_path or "",
            }
        )

    primary_np = np.stack(primary_images).astype(np.float32)
    secondary_np = np.stack(secondary_images).astype(np.float32)
    masks_np = np.stack(masks).astype(np.uint8)
    manifest = pd.DataFrame(rows)
    manifest["group_id"] = normalize_group_id_series(manifest["group_id"])

    if cache_dir is not None:
        np.save(primary_path, primary_np)
        np.save(secondary_path, secondary_np)
        np.save(masks_path, masks_np)
        manifest.to_csv(manifest_path, index=False)

    return primary_np, secondary_np, masks_np, manifest


def choose_validation_groups(
    manifest: pd.DataFrame,
    num_val_groups: int,
    explicit_groups: Optional[Sequence[str]],
    seed: int,
    min_auto_val_group_size: int,
) -> List[str]:
    manifest = manifest.copy()
    manifest["group_id"] = normalize_group_id_series(manifest["group_id"])
    available = sorted(manifest["group_id"].unique().tolist())
    if explicit_groups:
        explicit = [str(x).zfill(3) for x in explicit_groups]
        missing = sorted(set(explicit) - set(available))
        if missing:
            raise ValueError(f"Requested validation groups are missing: {missing}")
        if len(explicit) != num_val_groups:
            raise ValueError(f"Expected {num_val_groups} validation groups, got {len(explicit)}")
        return explicit

    group_sizes = manifest.groupby("group_id").size()
    group_sizes.index = normalize_group_id_series(group_sizes.index.to_series())
    group_sizes = group_sizes.groupby(level=0).sum().reindex(available, fill_value=0)
    eligible = sorted(group_sizes[group_sizes >= min_auto_val_group_size].index.tolist())
    if len(eligible) < num_val_groups:
        eligible = available
    if len(eligible) < num_val_groups:
        raise ValueError("Not enough groups to hold out validation samples.")
    return sorted(random.Random(seed).sample(eligible, num_val_groups))


class QPIDataset(Dataset):
    def __init__(
        self,
        primary_images: np.ndarray,
        secondary_images: np.ndarray,
        masks: np.ndarray,
        indices: Sequence[int],
        sample_ids: Sequence[str],
        patch_size: Optional[int],
        train: bool,
    ) -> None:
        self.primary_images = primary_images
        self.secondary_images = secondary_images
        self.masks = masks
        self.indices = np.array(indices)
        self.sample_ids = np.array(sample_ids)
        self.patch_size = patch_size
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def random_crop(self, primary: np.ndarray, secondary: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.patch_size is None or primary.shape[0] <= self.patch_size:
            return primary, secondary, mask
        top = np.random.randint(0, primary.shape[0] - self.patch_size + 1)
        left = np.random.randint(0, primary.shape[1] - self.patch_size + 1)
        return (
            primary[top : top + self.patch_size, left : left + self.patch_size],
            secondary[top : top + self.patch_size, left : left + self.patch_size],
            mask[top : top + self.patch_size, left : left + self.patch_size],
        )

    def augment(self, primary: np.ndarray, secondary: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            primary = np.flip(primary, axis=1).copy()
            secondary = np.flip(secondary, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            primary = np.flip(primary, axis=0).copy()
            secondary = np.flip(secondary, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        rotations = np.random.randint(0, 4)
        if rotations:
            primary = np.rot90(primary, rotations).copy()
            secondary = np.rot90(secondary, rotations).copy()
            mask = np.rot90(mask, rotations).copy()
        return primary, secondary, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample_index = self.indices[idx]
        primary = self.primary_images[sample_index]
        secondary = self.secondary_images[sample_index]
        mask = self.masks[sample_index]

        if self.train:
            primary, secondary, mask = self.random_crop(primary, secondary, mask)
            primary, secondary, mask = self.augment(primary, secondary, mask)

        return {
            "image": torch.from_numpy(primary[None]).float(),
            "secondary_image": torch.from_numpy(secondary[None]).float(),
            "mask": torch.from_numpy(mask).long(),
            "sample_id": str(self.sample_ids[sample_index]),
        }


def build_dataloaders(
    primary_images: np.ndarray,
    secondary_images: np.ndarray,
    masks: np.ndarray,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, List[str], np.ndarray, np.ndarray]:
    val_groups = choose_validation_groups(
        manifest=manifest,
        num_val_groups=args.num_val_groups,
        explicit_groups=args.val_groups,
        seed=args.seed,
        min_auto_val_group_size=args.min_auto_val_group_size,
    )
    groups = manifest["group_id"].astype(str).to_numpy()
    train_indices = np.where(~np.isin(groups, np.array(val_groups)))[0]
    val_indices = np.where(np.isin(groups, np.array(val_groups)))[0]

    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    train_dataset = QPIDataset(
        primary_images,
        secondary_images,
        masks,
        train_indices,
        sample_ids,
        patch_size=args.patch_size,
        train=True,
    )
    val_dataset = QPIDataset(
        primary_images,
        secondary_images,
        masks,
        val_indices,
        sample_ids,
        patch_size=None,
        train=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    return train_loader, val_loader, val_groups, train_indices, val_indices


def make_color_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color[mask == 1] = np.array([0, 255, 0], dtype=np.uint8)
    color[mask == 2] = np.array([255, 255, 0], dtype=np.uint8)
    return color


def compute_multiclass_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    ious: List[float] = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(1.0 if union == 0 else intersection / union)
    return float(np.mean(ious))


def compute_multiclass_dice(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    dices: List[float] = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        denom = pred_cls.sum().item() + target_cls.sum().item()
        dices.append(1.0 if denom == 0 else (2.0 * intersection) / denom)
    return float(np.mean(dices))


def compute_multiclass_precision_recall_f1(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> Tuple[float, float, float]:
    pred = logits.argmax(dim=1)
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        tp = (pred_cls & target_cls).sum().item()
        fp = (pred_cls & ~target_cls).sum().item()
        fn = (~pred_cls & target_cls).sum().item()

        precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 1.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        denom = torch.sum(probs + target_one_hot, dims)
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()
        return ce + dice


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = ConvNormAct(in_channels, out_channels)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return x, self.pool(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvNormAct(out_channels + skip_channels, out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class GilResidualUNet(nn.Module):
    """
    Close paper-inspired implementation of the modified residual U-Net from Gil et al. 2023.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = EncoderStage(in_channels, base_channels)
        self.enc2 = EncoderStage(base_channels, base_channels * 2)
        self.enc3 = EncoderStage(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderStage(base_channels * 4, base_channels * 8)
        self.bottleneck = nn.Sequential(
            ConvNormAct(base_channels * 8, base_channels * 16),
            ResidualBlock(base_channels * 16),
            ResidualBlock(base_channels * 16),
        )
        self.dec4 = DecoderStage(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderStage(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderStage(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderStage(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor, secondary_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        s1, x = self.enc1(image)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


class FusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(channels * 3, channels),
            ResidualBlock(channels),
        )

    def forward(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([primary, secondary, torch.abs(primary - secondary)], dim=1))


class SharedEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = EncoderStage(in_channels, base_channels)
        self.enc2 = EncoderStage(base_channels, base_channels * 2)
        self.enc3 = EncoderStage(base_channels * 2, base_channels * 4)
        self.enc4 = EncoderStage(base_channels * 4, base_channels * 8)
        self.bottleneck = nn.Sequential(
            ConvNormAct(base_channels * 8, base_channels * 16),
            ResidualBlock(base_channels * 16),
            ResidualBlock(base_channels * 16),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        b = self.bottleneck(x)
        return s1, s2, s3, s4, b


class DMQPMNet(nn.Module):
    """
    Practical dual-encoder fusion network inspired by DM-QPMNet.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        self.primary_encoder = SharedEncoder(in_channels, base_channels=base_channels)
        self.secondary_encoder = SharedEncoder(in_channels, base_channels=base_channels)

        self.fuse1 = FusionBlock(base_channels)
        self.fuse2 = FusionBlock(base_channels * 2)
        self.fuse3 = FusionBlock(base_channels * 4)
        self.fuse4 = FusionBlock(base_channels * 8)
        self.fuseb = FusionBlock(base_channels * 16)

        self.dec4 = DecoderStage(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderStage(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderStage(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderStage(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor, secondary_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        if secondary_image is None:
            raise ValueError("DM-QPMNet requires a secondary_image tensor.")

        p1, p2, p3, p4, pb = self.primary_encoder(image)
        s1, s2, s3, s4, sb = self.secondary_encoder(secondary_image)

        f1 = self.fuse1(p1, s1)
        f2 = self.fuse2(p2, s2)
        f3 = self.fuse3(p3, s3)
        f4 = self.fuse4(p4, s4)
        fb = self.fuseb(pb, sb)

        x = self.dec4(fb, f4)
        x = self.dec3(x, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        return self.head(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=device).float() / max(half_dim - 1, 1)
        emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeConditionedBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = ConvNormAct(out_channels, out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        scale = self.time_proj(t_emb)[:, :, None, None]
        x = x + scale
        x = self.conv2(x)
        return x


class PriorGuidedDiffSegLite(nn.Module):
    """
    Compact prior-guided diffusion-style segmenter inspired by PG-DiffSeg.
    """

    def __init__(self, image_channels: int = 1, num_classes: int = 3, base_channels: int = 32, timesteps: int = 50) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.time_dim = base_channels * 4

        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.time_dim, self.time_dim),
        )
        self.prior_encoder = nn.Sequential(
            ConvNormAct(image_channels, base_channels),
            ResidualBlock(base_channels),
            ConvNormAct(base_channels, base_channels),
            ResidualBlock(base_channels),
        )
        self.prior_to_noise = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        self.denoiser = nn.Sequential(
            ConvNormAct(num_classes + image_channels + base_channels, base_channels * 2),
            ResidualBlock(base_channels * 2),
            ResidualBlock(base_channels * 2),
            ConvNormAct(base_channels * 2, base_channels),
            ResidualBlock(base_channels),
            nn.Conv2d(base_channels, num_classes, kernel_size=1),
        )
        self.time_to_channels = nn.Linear(self.time_dim, base_channels * 2)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_noise(self, image: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        prior = self.prior_encoder(image)
        t_emb = self.time_embedding(t)
        t_proj = self.time_to_channels(t_emb)[:, :, None, None]
        denoise_input = torch.cat([xt, image, prior], dim=1)
        h = self.denoiser[0](denoise_input)
        h = h + t_proj
        for layer in self.denoiser[1:]:
            h = layer(h)
        return h

    def forward_loss(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        noise = torch.randn_like(target_one_hot)
        t = torch.randint(0, self.timesteps, (image.shape[0],), device=image.device)
        xt = self.q_sample(target_one_hot, t, noise)
        predicted_noise = self.predict_noise(image, xt, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, image: torch.Tensor, sample_steps: int) -> torch.Tensor:
        prior = self.prior_encoder(image)
        prior_noise = torch.tanh(self.prior_to_noise(prior))
        x = 0.7 * torch.randn(
            image.shape[0],
            self.num_classes,
            image.shape[2],
            image.shape[3],
            device=image.device,
        ) + 0.3 * prior_noise

        indices = torch.linspace(self.timesteps - 1, 0, sample_steps, device=image.device).long().unique(sorted=True)
        for t_scalar in reversed(indices.tolist()):
            t = torch.full((image.shape[0],), t_scalar, device=image.device, dtype=torch.long)
            predicted_noise = self.predict_noise(image, x, t)
            alpha_bar_t = self.alpha_bars[t][:, None, None, None]
            x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            if t_scalar > 0:
                prev = max(t_scalar - 1, 0)
                alpha_bar_prev = self.alpha_bars[prev].view(1, 1, 1, 1)
                x = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * predicted_noise
            else:
                x = x0
        return x


def create_model(model_name: str, args: argparse.Namespace) -> nn.Module:
    if model_name == "gil2023":
        return GilResidualUNet(num_classes=args.num_classes)
    if model_name == "dm_qpmnet":
        return DMQPMNet(num_classes=args.num_classes)
    if model_name == "pg_diffseg":
        return PriorGuidedDiffSegLite(num_classes=args.num_classes, timesteps=args.diffusion_steps)
    raise ValueError(f"Unsupported model: {model_name}")


def create_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def evaluate_standard_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int],
    save_dir: Optional[Path] = None,
) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            image = batch["image"].to(device)
            secondary = batch["secondary_image"].to(device)
            target = batch["mask"].to(device)
            logits = model(image, secondary if isinstance(model, DMQPMNet) else None)
            loss = criterion(logits, target)
            precision, recall, f1 = compute_multiclass_precision_recall_f1(logits, target, num_classes)
            total_loss += loss.item()
            total_iou += compute_multiclass_iou(logits, target, num_classes)
            total_dice += compute_multiclass_dice(logits, target, num_classes)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_batches += 1

            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                pred = logits.argmax(dim=1).cpu().numpy()
                for i, sample_id in enumerate(batch["sample_id"]):
                    Image.fromarray(make_color_mask(pred[i])).save(save_dir / f"{sample_id}_pred.png")
                    Image.fromarray(make_color_mask(target[i].cpu().numpy())).save(save_dir / f"{sample_id}_gt.png")

    denom = max(1, num_batches)
    return (
        total_loss / denom,
        total_iou / denom,
        total_dice / denom,
        total_precision / denom,
        total_recall / denom,
        total_f1 / denom,
    )


def evaluate_diffusion_model(
    model: PriorGuidedDiffSegLite,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int],
    sample_steps: int,
    save_dir: Optional[Path] = None,
) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            loss = model.forward_loss(image, target)
            logits = model.sample(image, sample_steps=sample_steps)
            precision, recall, f1 = compute_multiclass_precision_recall_f1(logits, target, num_classes)
            total_loss += loss.item()
            total_iou += compute_multiclass_iou(logits, target, num_classes)
            total_dice += compute_multiclass_dice(logits, target, num_classes)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_batches += 1

            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                pred = logits.argmax(dim=1).cpu().numpy()
                for i, sample_id in enumerate(batch["sample_id"]):
                    Image.fromarray(make_color_mask(pred[i])).save(save_dir / f"{sample_id}_pred.png")
                    Image.fromarray(make_color_mask(target[i].cpu().numpy())).save(save_dir / f"{sample_id}_gt.png")

    denom = max(1, num_batches)
    return (
        total_loss / denom,
        total_iou / denom,
        total_dice / denom,
        total_precision / denom,
        total_recall / denom,
        total_f1 / denom,
    )


def save_history(history: List[EpochMetrics], path: Path) -> None:
    pd.DataFrame([asdict(x) for x in history]).to_csv(path, index=False)


def train_single_model(
    model_name: str,
    primary_images: np.ndarray,
    secondary_images: np.ndarray,
    masks: np.ndarray,
    manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    model_cfg = choose_model_defaults(model_name, args)
    run_dir = args.output_root / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "split_manifest.csv").open("w", encoding="utf-8") as _:
        pass

    train_loader, val_loader, val_groups, train_indices, val_indices = build_dataloaders(
        primary_images=primary_images,
        secondary_images=secondary_images,
        masks=masks,
        manifest=manifest,
        args=args,
        batch_size=int(model_cfg["batch_size"]),
    )

    split_manifest = manifest.copy()
    split_manifest["split"] = "train"
    split_manifest.loc[val_indices, "split"] = "val"
    split_manifest.to_csv(run_dir / "split_manifest.csv", index=False)

    device = torch.device(args.device)
    model = create_model(model_name, args).to(device)
    optimizer = create_optimizer(
        model=model,
        optimizer_name=str(model_cfg["optimizer"]),
        lr=float(model_cfg["lr"]),
        weight_decay=float(args.weight_decay),
    )
    criterion = DiceCrossEntropyLoss(args.num_classes)

    history: List[EpochMetrics] = []
    best_miou = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(model_cfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

            image = batch["image"].to(device)
            secondary = batch["secondary_image"].to(device)
            target = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            if model_name == "pg_diffseg":
                loss = model.forward_loss(image, target)
            else:
                logits = model(image, secondary if model_name == "dm_qpmnet" else None)
                loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        train_loss = running_loss / max(1, batch_count)

        prediction_dir = run_dir / "val_predictions" if args.save_predictions and epoch == int(model_cfg["epochs"]) else None
        if model_name == "pg_diffseg":
            val_loss, val_miou, val_dice, val_precision, val_recall, val_f1 = evaluate_diffusion_model(
                model=model,
                loader=val_loader,
                device=device,
                num_classes=args.num_classes,
                max_batches=args.max_val_batches,
                sample_steps=args.diffusion_sample_steps,
                save_dir=prediction_dir,
            )
        else:
            val_loss, val_miou, val_dice, val_precision, val_recall, val_f1 = evaluate_standard_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=args.num_classes,
                max_batches=args.max_val_batches,
                save_dir=prediction_dir,
            )

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                val_miou=float(val_miou),
                val_dice=float(val_dice),
                val_precision=float(val_precision),
                val_recall=float(val_recall),
                val_f1=float(val_f1),
            )
        )

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_miou": val_miou,
                    "val_groups": val_groups,
                },
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"[{model_name}] epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_miou={val_miou:.4f} "
            f"val_dice={val_dice:.4f} "
            f"val_precision={val_precision:.4f} "
            f"val_recall={val_recall:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"[{model_name}] early stopping at epoch {epoch}")
            break

    save_history(history, run_dir / "history.csv")
    best_history = next((x for x in history if x.epoch == best_epoch), None)
    metadata = {
        "model_name": model_name,
        "val_groups": val_groups,
        "train_group_count": int(manifest.iloc[train_indices]["group_id"].nunique()),
        "val_group_count": int(manifest.iloc[val_indices]["group_id"].nunique()),
        "train_image_count": int(len(train_indices)),
        "val_image_count": int(len(val_indices)),
        "best_epoch": int(best_epoch),
        "best_val_miou": float(best_miou),
        "best_val_dice": float(best_history.val_dice if best_history else 0.0),
        "best_val_precision": float(best_history.val_precision if best_history else 0.0),
        "best_val_recall": float(best_history.val_recall if best_history else 0.0),
        "best_val_f1": float(best_history.val_f1 if best_history else 0.0),
        "paper_note": paper_note(model_name),
        "config": {
            "epochs": int(model_cfg["epochs"]),
            "batch_size": int(model_cfg["batch_size"]),
            "lr": float(model_cfg["lr"]),
            "optimizer": str(model_cfg["optimizer"]),
            "image_size": int(args.image_size),
            "patch_size": int(args.patch_size),
            "seed": int(args.seed),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def paper_note(model_name: str) -> str:
    notes = {
        "gil2023": "Residual U-Net close to the architecture and optimizer described in the paper.",
        "dm_qpmnet": "Dual-modality fusion approximation; if no second modality exists locally, the script duplicates the primary QPI image.",
        "pg_diffseg": "Compact prior-guided diffusion-style approximation rather than an exact latent-diffusion reproduction.",
        "cellsnap2024": "CellSNAP is a 3D rule-based segmentation algorithm and is not retrained here.",
        "chen2024": "Chen et al. 2024 is a review article, not a trainable model definition.",
    }
    return notes[model_name]


def handle_non_trainable(model_name: str) -> Dict[str, object]:
    note = paper_note(model_name)
    print(f"[{model_name}] skipped: {note}")
    return {"model_name": model_name, "skipped": True, "reason": note}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(
        data_root=args.data_root,
        secondary_image_pattern=args.secondary_image_pattern,
        dm_secondary_mode=args.dm_secondary_mode,
    )
    cache_dir = args.cache_dir or (args.output_root / "cache")
    primary_images, secondary_images, masks, manifest = preprocess_samples(
        samples=samples,
        image_size=args.image_size,
        require_secondary=args.dm_secondary_mode == "required",
        cache_dir=cache_dir,
        rebuild_cache=args.rebuild_cache,
    )

    print(f"Loaded {len(manifest)} image/mask pairs from {manifest['group_id'].nunique()} top-level groups.")
    print("Group counts:")
    print(manifest.groupby("group_id").size().to_string())

    results: List[Dict[str, object]] = []
    for model_name in args.models:
        if model_name in NON_TRAINABLE_MODELS:
            results.append(handle_non_trainable(model_name))
            continue
        results.append(
            train_single_model(
                model_name=model_name,
                primary_images=primary_images,
                secondary_images=secondary_images,
                masks=masks,
                manifest=manifest,
                args=args,
            )
        )

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
