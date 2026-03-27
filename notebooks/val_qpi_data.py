#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from train_qpi_paper_models import DMQPMNet, GilResidualUNet, PriorGuidedDiffSegLite


TRAINABLE_MODELS = ("our_unet", "gil2023", "dm_qpmnet", "pg_diffseg")


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
    parser = argparse.ArgumentParser(description="Train and validate notebook model plus QPI baselines.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/anzhe/home/labelled"))
    parser.add_argument("--output-root", type=Path, default=Path("./runs/qpi_binary_models"))
    parser.add_argument("--models", nargs="+", default=list(TRAINABLE_MODELS), choices=TRAINABLE_MODELS)
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
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--diffusion-sample-steps", type=int, default=25)
    parser.add_argument("--dm-secondary-mode", choices=("duplicate", "required"), default="duplicate")
    parser.add_argument("--secondary-image-pattern", type=str, default=None)
    parser.add_argument("--split-mode", choices=("sample", "group"), default="sample")
    parser.add_argument("--num-val-samples", type=int, default=2)
    parser.add_argument("--val-samples", nargs="*", default=None)
    parser.add_argument("--val-groups", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-num-classes",
        type=int,
        default=None,
        help="Number of output classes in the saved checkpoint. Use 3 for old live/dead/background weights.",
    )
    parser.add_argument(
        "--collapse-foreground",
        action="store_true",
        help="Collapse checkpoint classes 1 and 2 into foreground 1 during evaluation.",
    )
    return parser.parse_args()


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
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = torch.sum(probs * one_hot, dims)
        denom = torch.sum(probs + one_hot, dims)
        dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        return ce + dice


class SelfAttention(nn.Module):
    def __init__(self, channels: int, size: int) -> None:
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, residual: bool = False) -> None:
        super().__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return F.gelu(x + self.conv_block(x))
        return self.conv_block(x)


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, residual=True),
            ConvBlock(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([skip_x, x], dim=1)
        return self.conv(x)


class NotebookUNet(nn.Module):
    def __init__(self, c_in: int = 1, c_out: int = 2) -> None:
        super().__init__()
        self.initial_conv = ConvBlock(c_in, 64)
        self.downsample1 = DownSample(64, 128)
        self.self_attention1 = SelfAttention(128, 128)
        self.downsample2 = DownSample(128, 256)
        self.self_attention2 = SelfAttention(256, 256)
        self.downsample3 = DownSample(256, 256)
        self.self_attention3 = SelfAttention(256, 256)
        self.bottom1 = ConvBlock(256, 512)
        self.bottom2 = ConvBlock(512, 512)
        self.bottom3 = ConvBlock(512, 256)
        self.upsample1 = UpSample(512, 128)
        self.self_attention4 = SelfAttention(128, 128)
        self.upsample2 = UpSample(256, 64)
        self.self_attention5 = SelfAttention(64, 64)
        self.upsample3 = UpSample(128, 64)
        self.self_attention6 = SelfAttention(64, 64)
        self.output_conv = nn.Conv2d(64, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor, secondary_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = self.initial_conv(x)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        x4 = self.downsample3(x3)
        x4 = self.bottom1(x4)
        x4 = self.bottom2(x4)
        x4 = self.bottom3(x4)
        x = self.upsample1(x4, x3)
        x = self.upsample2(x, x2)
        x = self.upsample3(x, x1)
        return self.output_conv(x)


def model_defaults(model_name: str, args: argparse.Namespace) -> Dict[str, float | int | str]:
    defaults: Dict[str, Dict[str, float | int | str]] = {
        "our_unet": {"epochs": 200, "batch_size": 5, "lr": 1e-4, "optimizer": "sgd"},
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


def build_model(model_name: str, args: argparse.Namespace, num_classes: Optional[int] = None) -> nn.Module:
    out_classes = args.num_classes if num_classes is None else num_classes
    if model_name == "our_unet":
        return NotebookUNet(c_in=1, c_out=out_classes)
    if model_name == "gil2023":
        return GilResidualUNet(in_channels=1, num_classes=out_classes)
    if model_name == "dm_qpmnet":
        return DMQPMNet(in_channels=1, num_classes=out_classes)
    if model_name == "pg_diffseg":
        return PriorGuidedDiffSegLite(image_channels=1, num_classes=out_classes, timesteps=args.diffusion_steps)
    raise ValueError(model_name)


def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(optimizer_name)


def make_mask_png(mask: np.ndarray) -> np.ndarray:
    if mask.max() <= 1:
        return (mask.astype(np.uint8) * 255)
    return mask.astype(np.uint8)


def collapse_logits_to_binary(logits: torch.Tensor) -> torch.Tensor:
    """
    Collapse multi-class logits to binary logits:
    class 0 -> background
    classes 1..N-1 -> foreground
    """
    if logits.shape[1] == 2:
        return logits
    probs = torch.softmax(logits, dim=1)
    bg = probs[:, :1]
    fg = probs[:, 1:].sum(dim=1, keepdim=True)
    collapsed = torch.cat([bg, fg], dim=1).clamp_min(1e-8)
    return torch.log(collapsed)


def load_checkpoint_into_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def evaluate_checkpoint_binary(
    model_name: str,
    manifest: pd.DataFrame,
    val_indices: np.ndarray,
    split_meta: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    if args.checkpoint_path is None:
        raise ValueError("--checkpoint-path is required with --eval-only")

    checkpoint_num_classes = args.checkpoint_num_classes or args.num_classes
    run_dir = args.output_root / f"{model_name}_eval_only"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model(model_name, args, num_classes=checkpoint_num_classes).to(device)
    checkpoint_meta = load_checkpoint_into_model(model, args.checkpoint_path, device)

    val_dataset = ProcessedQPIDataset(manifest, val_indices, patch_size=args.image_size, train=False)
    cfg = model_defaults(model_name, args)
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(cfg["batch_size"]) // 2),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if args.max_val_batches is not None and batch_idx >= args.max_val_batches:
                break
            image = batch["image"].to(device)
            secondary = batch["secondary_image"].to(device)
            target = batch["mask"].to(device)

            if model_name == "pg_diffseg":
                logits = model.sample(image, sample_steps=args.diffusion_sample_steps)
            else:
                logits = model(image, secondary if model_name == "dm_qpmnet" else None)

            if args.collapse_foreground:
                logits = collapse_logits_to_binary(logits)

            total_iou += compute_multiclass_iou(logits, target, 2)
            total_dice += compute_multiclass_dice(logits, target, 2)
            precision, recall, f1 = compute_multiclass_precision_recall_f1(logits, target, 2)
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            num_batches += 1

            if args.save_predictions:
                pred_dir = run_dir / "val_predictions"
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred = logits.argmax(dim=1).cpu().numpy()
                for i, sample_id in enumerate(batch["sample_id"]):
                    Image.fromarray(make_mask_png(pred[i])).save(pred_dir / f"{sample_id}_pred.png")
                    Image.fromarray(make_mask_png(target[i].cpu().numpy())).save(pred_dir / f"{sample_id}_gt.png")

    denom = max(1, num_batches)
    summary = {
        "model_name": model_name,
        "mode": "eval_only",
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_num_classes": int(checkpoint_num_classes),
        "collapse_foreground": bool(args.collapse_foreground),
        "val_samples": split_meta["val_samples"],
        "val_groups": split_meta["val_groups"],
        "val_miou": total_iou / denom,
        "val_dice": total_dice / denom,
        "val_precision": total_precision / denom,
        "val_recall": total_recall / denom,
        "val_f1": total_f1 / denom,
        "checkpoint_meta_keys": sorted(list(checkpoint_meta.keys())) if isinstance(checkpoint_meta, dict) else [],
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_standard(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int],
    save_dir: Optional[Path],
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
                    Image.fromarray(make_mask_png(pred[i])).save(save_dir / f"{sample_id}_pred.png")
                    Image.fromarray(make_mask_png(target[i].cpu().numpy())).save(save_dir / f"{sample_id}_gt.png")

    denom = max(1, num_batches)
    return (
        total_loss / denom,
        total_iou / denom,
        total_dice / denom,
        total_precision / denom,
        total_recall / denom,
        total_f1 / denom,
    )


def evaluate_diffusion(
    model: PriorGuidedDiffSegLite,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: Optional[int],
    sample_steps: int,
    save_dir: Optional[Path],
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
                    Image.fromarray(make_mask_png(pred[i])).save(save_dir / f"{sample_id}_pred.png")
                    Image.fromarray(make_mask_png(target[i].cpu().numpy())).save(save_dir / f"{sample_id}_gt.png")

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
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    split_meta: Dict[str, List[str]],
) -> Dict[str, object]:
    cfg = model_defaults(model_name, args)
    run_dir = args.output_root / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_model(model_name, args).to(device)
    optimizer = build_optimizer(model, str(cfg["optimizer"]), float(cfg["lr"]), args.weight_decay)
    criterion = DiceCrossEntropyLoss(args.num_classes)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["epochs"])))

    train_dataset = ProcessedQPIDataset(manifest, train_indices, patch_size=args.patch_size, train=True)
    val_dataset = ProcessedQPIDataset(manifest, val_indices, patch_size=args.image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=max(1, int(cfg["batch_size"]) // 2), shuffle=False, num_workers=args.num_workers)

    split_df = manifest.copy()
    split_df["split"] = "train"
    split_df.loc[val_indices, "split"] = "val"
    split_df.to_csv(run_dir / "split_manifest.csv", index=False)

    history: List[EpochMetrics] = []
    best_miou = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
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

        scheduler.step()
        train_loss = running_loss / max(1, batch_count)

        save_dir = run_dir / "val_predictions" if args.save_predictions and epoch == int(cfg["epochs"]) else None
        if model_name == "pg_diffseg":
            val_loss, val_miou, val_dice, val_precision, val_recall, val_f1 = evaluate_diffusion(
                model,
                val_loader,
                device,
                args.num_classes,
                args.max_val_batches,
                args.diffusion_sample_steps,
                save_dir,
            )
        else:
            val_loss, val_miou, val_dice, val_precision, val_recall, val_f1 = evaluate_standard(
                model,
                val_loader,
                criterion,
                device,
                args.num_classes,
                args.max_val_batches,
                save_dir,
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
                    "split_meta": split_meta,
                },
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[{model_name}] early stopping at epoch {epoch}")
            break

    save_history(history, run_dir / "history.csv")
    best_history = next((x for x in history if x.epoch == best_epoch), None)
    metadata = {
        "model_name": model_name,
        "split_mode": args.split_mode,
        "val_samples": split_meta["val_samples"],
        "val_groups": split_meta["val_groups"],
        "train_sample_count": int(len(train_indices)),
        "val_sample_count": int(len(val_indices)),
        "best_epoch": int(best_epoch),
        "best_val_miou": float(best_miou),
        "best_val_dice": float(best_history.val_dice if best_history else 0.0),
        "best_val_precision": float(best_history.val_precision if best_history else 0.0),
        "best_val_recall": float(best_history.val_recall if best_history else 0.0),
        "best_val_f1": float(best_history.val_f1 if best_history else 0.0),
        "config": {
            "epochs": int(cfg["epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "optimizer": str(cfg["optimizer"]),
            "patch_size": int(args.patch_size),
            "image_size": int(args.image_size),
            "num_classes": int(args.num_classes),
            "seed": int(args.seed),
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = collect_samples(args.data_root, args.secondary_image_pattern, args.dm_secondary_mode)
    train_indices, val_indices, split_meta = split_manifest(manifest, args)

    print(f"Loaded {len(manifest)} processed samples from {args.data_root}")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Val groups: {split_meta['val_groups']}")
    print(f"Val samples: {split_meta['val_samples']}")

    results = []
    if args.eval_only:
        if len(args.models) != 1:
            raise ValueError("--eval-only currently requires exactly one model in --models")
        results.append(
            evaluate_checkpoint_binary(
                model_name=args.models[0],
                manifest=manifest,
                val_indices=val_indices,
                split_meta=split_meta,
                args=args,
            )
        )
        summary_path = args.output_root / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved summary to {summary_path}")
        return

    for model_name in args.models:
        results.append(train_single_model(model_name, manifest, args, train_indices, val_indices, split_meta))

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
