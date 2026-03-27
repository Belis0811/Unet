#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


MODELS = ("e_unet", "cellotype", "sam2_zeroshot", "micro_sam_zeroshot")


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.dtype == np.uint16:
        image = image / 65535.0
    elif image.max() > image.min():
        low, high = np.percentile(image, (1, 99))
        if high > low:
            image = np.clip((image - low) / (high - low), 0.0, 1.0)
        else:
            image = np.zeros_like(image, dtype=np.float32)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image.astype(np.float32)


def load_png_image(path: Path) -> np.ndarray:
    return normalize_image(np.array(Image.open(path)))


def load_binary_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    return (mask > 0).astype(np.uint8)


def collect_samples(root: Path, secondary_image_pattern: Optional[str], dm_secondary_mode: str) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for group_dir in sorted(x for x in root.iterdir() if x.is_dir()):
        for sample_dir in sorted(x for x in group_dir.iterdir() if x.is_dir()):
            images_dir = sample_dir / "images"
            masks_dir = sample_dir / "masks"
            if not images_dir.is_dir() or not masks_dir.is_dir():
                continue
            image_candidates = sorted(images_dir.glob("*_mSLIM.png"))
            mask_candidates = sorted(masks_dir.glob("*.png"))
            if not image_candidates or not mask_candidates:
                continue

            secondary_path = None
            if secondary_image_pattern:
                sec = sorted(images_dir.glob(secondary_image_pattern))
                if sec:
                    secondary_path = str(sec[0])
            elif dm_secondary_mode == "duplicate":
                secondary_path = str(image_candidates[0])

            rows.append(
                {
                    "group_id": group_dir.name,
                    "sample_id": sample_dir.name,
                    "sample_dir": str(sample_dir),
                    "image_path": str(image_candidates[0]),
                    "mask_path": str(mask_candidates[0]),
                    "secondary_path": secondary_path or "",
                }
            )
    if not rows:
        raise FileNotFoundError(f"No processed samples found under {root}")
    return pd.DataFrame(rows)


class ProcessedQPIDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, indices: Sequence[int], patch_size: Optional[int], train: bool) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.indices = np.array(indices)
        self.patch_size = patch_size
        self.train = train

    def __len__(self) -> int:
        return len(self.indices)

    def random_crop(self, image: np.ndarray, secondary: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.patch_size is None or image.shape[0] <= self.patch_size or image.shape[1] <= self.patch_size:
            return image, secondary, mask
        top = np.random.randint(0, image.shape[0] - self.patch_size + 1)
        left = np.random.randint(0, image.shape[1] - self.patch_size + 1)
        sl = np.s_[top : top + self.patch_size, left : left + self.patch_size]
        return image[sl], secondary[sl], mask[sl]

    def augment(self, image: np.ndarray, secondary: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            secondary = np.flip(secondary, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            secondary = np.flip(secondary, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        rotations = np.random.randint(0, 4)
        if rotations:
            image = np.rot90(image, rotations).copy()
            secondary = np.rot90(secondary, rotations).copy()
            mask = np.rot90(mask, rotations).copy()
        return image, secondary, mask

    def resize_if_needed(self, image: np.ndarray, secondary: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return image, secondary, mask
        if image.shape[0] == self.patch_size and image.shape[1] == self.patch_size:
            return image, secondary, mask
        image_t = torch.from_numpy(image)[None, None].float()
        sec_t = torch.from_numpy(secondary)[None, None].float()
        mask_t = torch.from_numpy(mask)[None, None].float()
        image_r = F.interpolate(image_t, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)[0, 0].numpy()
        sec_r = F.interpolate(sec_t, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False)[0, 0].numpy()
        mask_r = F.interpolate(mask_t, size=(self.patch_size, self.patch_size), mode="nearest")[0, 0].numpy().astype(np.uint8)
        return image_r, sec_r, mask_r

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[self.indices[idx]]
        image = load_png_image(Path(row["image_path"]))
        secondary = load_png_image(Path(row["secondary_path"])) if row["secondary_path"] else image.copy()
        mask = load_binary_mask(Path(row["mask_path"]))

        image, secondary, mask = self.resize_if_needed(image, secondary, mask)
        if self.train:
            image, secondary, mask = self.random_crop(image, secondary, mask)
            image, secondary, mask = self.augment(image, secondary, mask)

        return {
            "image": torch.from_numpy(image[None]).float(),
            "secondary_image": torch.from_numpy(secondary[None]).float(),
            "mask": torch.from_numpy(mask).long(),
            "sample_id": str(row["sample_id"]),
            "group_id": str(row["group_id"]),
        }


def split_manifest(manifest: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    if args.split_mode == "sample":
        available = manifest["sample_id"].astype(str).tolist()
        if args.val_samples:
            val_samples = [str(x) for x in args.val_samples]
        else:
            if len(available) <= args.num_val_samples:
                raise ValueError("Not enough samples for the requested validation split.")
            val_samples = sorted(random.Random(args.seed).sample(available, args.num_val_samples))
        train_indices = np.where(~manifest["sample_id"].astype(str).isin(val_samples))[0]
        val_indices = np.where(manifest["sample_id"].astype(str).isin(val_samples))[0]
        meta = {"val_samples": val_samples, "val_groups": sorted(manifest.iloc[val_indices]["group_id"].astype(str).unique().tolist())}
        return train_indices, val_indices, meta

    available_groups = sorted(manifest["group_id"].astype(str).unique().tolist())
    if args.val_groups:
        val_groups = [str(x) for x in args.val_groups]
    else:
        if len(available_groups) <= args.num_val_samples:
            raise ValueError("Not enough groups for the requested validation split.")
        val_groups = sorted(random.Random(args.seed).sample(available_groups, args.num_val_samples))
    train_indices = np.where(~manifest["group_id"].astype(str).isin(val_groups))[0]
    val_indices = np.where(manifest["group_id"].astype(str).isin(val_groups))[0]
    meta = {"val_samples": manifest.iloc[val_indices]["sample_id"].astype(str).tolist(), "val_groups": val_groups}
    return train_indices, val_indices, meta


def compute_multiclass_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    values: List[float] = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        inter = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        values.append(1.0 if union == 0 else inter / union)
    return float(np.mean(values))


def compute_multiclass_dice(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=1)
    values: List[float] = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        inter = (pred_cls & target_cls).sum().item()
        denom = pred_cls.sum().item() + target_cls.sum().item()
        values.append(1.0 if denom == 0 else (2.0 * inter) / denom)
    return float(np.mean(values))


def compute_multiclass_precision_recall_f1(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Tuple[float, float, float]:
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
        one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dims)
        denom = torch.sum(probs + one_hot, dims)
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        return ce + dice


def make_mask_png(mask: np.ndarray) -> np.ndarray:
    if mask.max() <= 1:
        return mask.astype(np.uint8) * 255
    return mask.astype(np.uint8)


def save_history(history: List[EpochMetrics], path: Path) -> None:
    pd.DataFrame([asdict(x) for x in history]).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate additional QPI segmentation models.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/anzhe/home/labelled"))
    parser.add_argument("--output-root", type=Path, default=Path("./runs/additional_qpi_models"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--split-mode", choices=("sample", "group"), default="sample")
    parser.add_argument("--num-val-samples", type=int, default=2)
    parser.add_argument("--val-samples", nargs="*", default=None)
    parser.add_argument("--val-groups", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--secondary-image-pattern", type=str, default=None)
    parser.add_argument("--dm-secondary-mode", choices=("duplicate", "required"), default="duplicate")

    parser.add_argument("--sam2-config", type=str, default=None)
    parser.add_argument("--sam2-checkpoint", type=str, default=None)
    parser.add_argument("--micro-sam-model-type", type=str, default="vit_b_lm")
    parser.add_argument("--micro-sam-checkpoint", type=str, default=None)
    parser.add_argument("--prompt-mode", choices=("box", "box_and_point"), default="box")

    parser.add_argument("--cellotype-config", type=str, default=None)
    parser.add_argument("--cellotype-weights", type=str, default=None)
    parser.add_argument("--cellotype-run-train", action="store_true")
    return parser.parse_args()


def compute_binary_mask_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    union = np.logical_or(pred, target).sum()
    denom = pred.sum() + target.sum()
    precision = 1.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 1.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 1.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    dice = 1.0 if denom == 0 else (2.0 * tp) / denom
    iou = 1.0 if union == 0 else tp / union
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def mask_to_center_point(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    return point_coords, point_labels


class InitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        main_channels = out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, main_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 4),
            nn.PReLU(channels // 4),
            nn.Conv2d(
                channels // 4,
                channels // 4,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels // 4),
            nn.PReLU(channels // 4),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            Bottleneck(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class EUNetLite(nn.Module):
    """
    Lightweight E-UNet-style approximation for binary semantic segmentation.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.init = InitialBlock(in_channels, 16)
        self.down1 = DownBlock(16, 64)
        self.b1 = nn.Sequential(Bottleneck(64), Bottleneck(64, dilation=2), Bottleneck(64, dilation=4))
        self.down2 = DownBlock(64, 128)
        self.b2 = nn.Sequential(Bottleneck(128), Bottleneck(128, dilation=2), Bottleneck(128, dilation=4))
        self.up1 = UpBlock(128, 64, 64)
        self.up2 = UpBlock(64, 16, 32)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.PReLU(16),
        )
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.init(x)
        s2 = self.b1(self.down1(s1))
        x = self.b2(self.down2(s2))
        x = self.up1(x, s2)
        x = self.up2(x, s1)
        x = self.up3(x)
        return self.head(x)


def train_eunet(
    manifest: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    split_meta: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    run_dir = args.output_root / "e_unet"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = EUNetLite(in_channels=1, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = DiceCrossEntropyLoss(args.num_classes)

    train_loader = DataLoader(
        ProcessedQPIDataset(manifest, train_indices, patch_size=args.patch_size, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ProcessedQPIDataset(manifest, val_indices, patch_size=args.image_size, train=False),
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.num_workers,
    )

    history: List[EpochMetrics] = []
    best_miou = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            train_batches += 1
        scheduler.step()

        model.eval()
        totals = dict(loss=0.0, iou=0.0, dice=0.0, precision=0.0, recall=0.0, f1=0.0)
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                    break
                image = batch["image"].to(device)
                target = batch["mask"].to(device)
                logits = model(image)
                loss = criterion(logits, target)
                precision, recall, f1 = compute_multiclass_precision_recall_f1(logits, target, args.num_classes)
                totals["loss"] += loss.item()
                totals["iou"] += compute_multiclass_iou(logits, target, args.num_classes)
                totals["dice"] += compute_multiclass_dice(logits, target, args.num_classes)
                totals["precision"] += precision
                totals["recall"] += recall
                totals["f1"] += f1
                val_batches += 1
                if args.save_predictions and epoch == args.epochs:
                    pred_dir = run_dir / "val_predictions"
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    for i, sample_id in enumerate(batch["sample_id"]):
                        Image.fromarray(make_mask_png(pred[i])).save(pred_dir / f"{sample_id}_pred.png")
                        Image.fromarray(make_mask_png(target[i].cpu().numpy())).save(pred_dir / f"{sample_id}_gt.png")

        denom = max(1, val_batches)
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss_total / max(1, train_batches),
            val_loss=totals["loss"] / denom,
            val_miou=totals["iou"] / denom,
            val_dice=totals["dice"] / denom,
            val_precision=totals["precision"] / denom,
            val_recall=totals["recall"] / denom,
            val_f1=totals["f1"] / denom,
        )
        history.append(epoch_metrics)
        print(
            f"[e_unet] epoch {epoch:03d} "
            f"train_loss={epoch_metrics.train_loss:.4f} "
            f"val_loss={epoch_metrics.val_loss:.4f} "
            f"val_miou={epoch_metrics.val_miou:.4f} "
            f"val_dice={epoch_metrics.val_dice:.4f} "
            f"val_precision={epoch_metrics.val_precision:.4f} "
            f"val_recall={epoch_metrics.val_recall:.4f} "
            f"val_f1={epoch_metrics.val_f1:.4f}"
        )
        if epoch_metrics.val_miou > best_miou:
            best_miou = epoch_metrics.val_miou
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, run_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    save_history(history, run_dir / "history.csv")
    best = next((x for x in history if x.epoch == best_epoch), None)
    metadata = {
        "model_name": "e_unet",
        "note": "Lightweight E-UNet-style approximation implemented locally.",
        "val_samples": split_meta["val_samples"],
        "val_groups": split_meta["val_groups"],
        "best_epoch": best_epoch,
        "best_val_miou": best.val_miou if best else 0.0,
        "best_val_dice": best.val_dice if best else 0.0,
        "best_val_precision": best.val_precision if best else 0.0,
        "best_val_recall": best.val_recall if best else 0.0,
        "best_val_f1": best.val_f1 if best else 0.0,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_sam2_zeroshot(manifest: pd.DataFrame, val_indices: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    if not args.sam2_config or not args.sam2_checkpoint:
        raise ValueError("--sam2-config and --sam2-checkpoint are required for sam2_zeroshot")
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError("sam2 is not installed in this environment.") from exc

    device = torch.device(args.device)
    predictor = SAM2ImagePredictor(build_sam2(args.sam2_config, args.sam2_checkpoint, device=device))
    run_dir = args.output_root / "sam2_zeroshot"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics: List[Dict[str, object]] = []

    for idx in val_indices:
        row = manifest.iloc[idx]
        image_gray = np.array(Image.open(row["image_path"]))
        image_rgb = np.repeat(((image_gray / max(image_gray.max(), 1)) * 255).astype(np.uint8)[..., None], 3, axis=2)
        target = (np.array(Image.open(row["mask_path"])) > 0).astype(np.uint8)
        box = mask_to_box(target)
        point_coords, point_labels = mask_to_center_point(target)

        with torch.inference_mode():
            predictor.set_image(image_rgb)
            if args.prompt_mode == "box_and_point":
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=False,
                )
            else:
                masks, scores, _ = predictor.predict(box=box, multimask_output=False)
        pred = masks[0].astype(np.uint8)
        sample_metrics = compute_binary_mask_metrics(pred, target)
        sample_metrics.update({"sample_id": row["sample_id"], "score": float(scores[0]) if len(scores) else 0.0})
        metrics.append(sample_metrics)
        if args.save_predictions:
            pred_dir = run_dir / "val_predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(make_mask_png(pred)).save(pred_dir / f"{row['sample_id']}_pred.png")
            Image.fromarray(make_mask_png(target)).save(pred_dir / f"{row['sample_id']}_gt.png")

    df = pd.DataFrame(metrics)
    df.to_csv(run_dir / "metrics.csv", index=False)
    summary = {"model_name": "sam2_zeroshot", "prompt_mode": args.prompt_mode, **df.mean(numeric_only=True).to_dict()}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_micro_sam_zeroshot(manifest: pd.DataFrame, val_indices: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    try:
        from micro_sam import util as micro_util
        from micro_sam import prompt_based_segmentation as prompt_seg
    except ImportError as exc:
        raise ImportError("micro-sam is not installed in this environment.") from exc

    predictor = micro_util.get_sam_model(
        model_type=args.micro_sam_model_type,
        checkpoint_path=args.micro_sam_checkpoint,
        device=args.device,
    )
    run_dir = args.output_root / "micro_sam_zeroshot"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics: List[Dict[str, object]] = []

    for idx in val_indices:
        row = manifest.iloc[idx]
        image_gray = np.array(Image.open(row["image_path"]))
        image_rgb = np.repeat(((image_gray / max(image_gray.max(), 1)) * 255).astype(np.uint8)[..., None], 3, axis=2)
        target = (np.array(Image.open(row["mask_path"])) > 0).astype(np.uint8)
        box = mask_to_box(target)
        embeddings = micro_util.precompute_image_embeddings(predictor, image_rgb)
        if args.prompt_mode == "box_and_point":
            point_coords, point_labels = mask_to_center_point(target)
            masks = prompt_seg.segment_from_box_and_points(
                predictor,
                box=box,
                points=point_coords,
                labels=point_labels,
                image_embeddings=embeddings,
                multimask_output=False,
            )
        else:
            masks = prompt_seg.segment_from_box(
                predictor,
                box=box,
                image_embeddings=embeddings,
                multimask_output=False,
            )
        pred = np.asarray(masks[0] if masks.ndim == 3 else masks).astype(np.uint8)
        sample_metrics = compute_binary_mask_metrics(pred, target)
        sample_metrics.update({"sample_id": row["sample_id"]})
        metrics.append(sample_metrics)
        if args.save_predictions:
            pred_dir = run_dir / "val_predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(make_mask_png(pred)).save(pred_dir / f"{row['sample_id']}_pred.png")
            Image.fromarray(make_mask_png(target)).save(pred_dir / f"{row['sample_id']}_gt.png")

    df = pd.DataFrame(metrics)
    df.to_csv(run_dir / "metrics.csv", index=False)
    summary = {"model_name": "micro_sam_zeroshot", "prompt_mode": args.prompt_mode, **df.mean(numeric_only=True).to_dict()}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def export_cellotype_dataset(manifest: pd.DataFrame, train_indices: np.ndarray, val_indices: np.ndarray, out_dir: Path) -> Path:
    from skimage.measure import find_contours, label

    data_dir = out_dir / "cellotype_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    def build_records(indices: Sequence[int], split: str) -> List[dict]:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records: List[dict] = []
        for local_idx, idx in enumerate(indices):
            row = manifest.iloc[idx]
            image = np.array(Image.open(row["image_path"])).astype(np.float32)
            image = (image / max(image.max(), 1.0) * 255.0).astype(np.uint8)
            image_rgb = np.repeat(image[..., None], 3, axis=2)
            file_name = split_dir / f"X_{local_idx}.png"
            Image.fromarray(image_rgb).save(file_name)

            mask = (np.array(Image.open(row["mask_path"])) > 0).astype(np.uint8)
            labeled = label(mask)
            annotations = []
            for obj_id in range(1, labeled.max() + 1):
                obj_mask = (labeled == obj_id).astype(np.uint8)
                ys, xs = np.where(obj_mask > 0)
                if len(xs) == 0:
                    continue
                contours = find_contours(obj_mask, 0.5)
                if not contours:
                    continue
                contour = max(contours, key=len)
                if len(contour) <= 5:
                    continue
                polygon = np.stack([contour[:, 1], contour[:, 0]], axis=1).reshape(-1).tolist()
                annotations.append(
                    {
                        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                        "bbox_mode": 0,
                        "category_id": 0,
                        "segmentation": [polygon],
                    }
                )
            records.append(
                {
                    "file_name": str(file_name),
                    "height": int(image_rgb.shape[0]),
                    "width": int(image_rgb.shape[1]),
                    "image_id": int(local_idx),
                    "annotations": annotations,
                }
            )
        return records

    train_records = build_records(train_indices, "train")
    val_records = build_records(val_indices, "val")
    np.save(data_dir / "dataset_dicts_cell_train.npy", train_records)
    np.save(data_dir / "dataset_dicts_cell_val.npy", val_records)
    np.save(data_dir / "dataset_dicts_cell_test.npy", val_records)
    return data_dir


def run_cellotype(manifest: pd.DataFrame, train_indices: np.ndarray, val_indices: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    run_dir = args.output_root / "cellotype"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = export_cellotype_dataset(manifest, train_indices, val_indices, run_dir)

    summary: Dict[str, object] = {
        "model_name": "cellotype",
        "note": "Exports QPI data to the Tissuenet-style CelloType format. Training runs only if the CelloType stack is installed.",
        "data_dir": str(data_dir),
    }

    if not args.cellotype_run_train:
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    if not args.cellotype_config or not args.cellotype_weights:
        raise ValueError("--cellotype-config and --cellotype-weights are required when --cellotype-run-train is set")

    try:
        from cellotype.trainer import (
            Trainer,
            add_deeplab_config,
            add_maskdino_config,
            get_cfg,
        )
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.checkpoint import DetectionCheckpointer
    except ImportError as exc:
        raise ImportError("CelloType / Detectron2 is not installed in this environment.") from exc

    for split in ("train", "val", "test"):
        name = f"cell_{split}"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        DatasetCatalog.register(name, lambda split=split: np.load(data_dir / f"dataset_dicts_cell_{split}.npy", allow_pickle=True))
        MetadataCatalog.get(name).set(thing_classes=["cell"])

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.cellotype_config)
    cfg.MODEL.WEIGHTS = args.cellotype_weights
    cfg.MODEL.IN_CHANS = 3
    cfg.DATASETS.TRAIN = ("cell_train",)
    cfg.DATASETS.VAL = ("cell_val",)
    cfg.DATASETS.TEST = ("cell_test",)
    cfg.OUTPUT_DIR = str(run_dir / "output")
    cfg.SOLVER.AMP.ENABLED = False

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    results = Trainer.test(cfg, trainer.model)
    DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR).save("cellotype_qpi_best")
    summary["results"] = results
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = collect_samples(args.data_root, args.secondary_image_pattern, args.dm_secondary_mode)
    train_indices, val_indices, split_meta = split_manifest(manifest, args)

    print(f"Loaded {len(manifest)} samples from {args.data_root}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")

    results = []
    for model_name in args.models:
        if model_name == "e_unet":
            results.append(train_eunet(manifest, train_indices, val_indices, split_meta, args))
        elif model_name == "sam2_zeroshot":
            results.append(run_sam2_zeroshot(manifest, val_indices, args))
        elif model_name == "micro_sam_zeroshot":
            results.append(run_micro_sam_zeroshot(manifest, val_indices, args))
        elif model_name == "cellotype":
            results.append(run_cellotype(manifest, train_indices, val_indices, args))
        else:
            raise ValueError(model_name)

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
