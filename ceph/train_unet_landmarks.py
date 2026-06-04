import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model_factory import build_model, load_checkpoint_state


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def draw_gaussian(heatmap: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    tmp_size = int(sigma * 3)
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)

    h, w = heatmap.shape
    ul = [mu_x - tmp_size, mu_y - tmp_size]
    br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]

    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, None]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)

    heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
        heatmap[img_y[0] : img_y[1], img_x[0] : img_x[1]],
        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
    )
    return heatmap


class CephLandmarkDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        annotator: str,
        target_landmarks: list[str],
        image_size=(512, 512),
        heatmap_size=(128, 128),
        sigma=2.5,
        num_joints=29,
        is_train=True,
        rot_deg=8.0,
        scale_range=0.08,
        intensity_jitter=0.10,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.annotator = annotator
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.num_joints = num_joints
        self.is_train = is_train
        self.rot_deg = rot_deg
        self.scale_range = scale_range
        self.intensity_jitter = intensity_jitter
        self.target_landmarks = target_landmarks

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.image_dir = self.data_root / split / "Cephalograms"
        self.ann_dir = self.data_root / split / "Annotations" / "Cephalometric Landmarks" / annotator

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.ann_dir.exists():
            raise FileNotFoundError(f"Annotation directory does not exist: {self.ann_dir}")

        valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in valid_ext])
        self.samples = []
        for img_path in image_paths:
            ann_path = self.ann_dir / f"{img_path.stem}.json"
            if ann_path.exists():
                self.samples.append((img_path, ann_path))

        if not self.samples:
            raise RuntimeError(f"No samples found in split={split}")

        with open(self.samples[0][1], "r", encoding="utf-8") as f:
            first_ann = json.load(f)
        available_symbols = [lm["symbol"] for lm in first_ann["landmarks"]]
        missing = [symbol for symbol in self.target_landmarks if symbol not in available_symbols]
        if missing:
            raise ValueError(f"Missing landmarks in annotations: {missing}")
        self.landmark_order = [symbol for symbol in self.target_landmarks if symbol in available_symbols]
        if len(self.landmark_order) != self.num_joints:
            raise ValueError(
                f"Expected {self.num_joints} joints but found {len(self.landmark_order)} after filtering"
            )

        self.pixel_size_map = {}
        mapping_csv = self.data_root / "cephalogram_machine_mappings.csv"
        if mapping_csv.exists():
            df = pd.read_csv(mapping_csv)
            if "cephalogram_id" in df.columns and "pixel_size" in df.columns:
                for _, row in df.iterrows():
                    self.pixel_size_map[str(row["cephalogram_id"])] = float(row["pixel_size"])

    def __len__(self):
        return len(self.samples)

    def _load_points(self, ann_path: Path) -> np.ndarray:
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        symbol_to_xy = {lm["symbol"]: (lm["value"]["x"], lm["value"]["y"]) for lm in ann["landmarks"]}
        pts = []
        for symbol in self.landmark_order:
            if symbol not in symbol_to_xy:
                raise KeyError(f"Missing symbol {symbol} in {ann_path}")
            pts.append(symbol_to_xy[symbol])
        return np.array(pts, dtype=np.float32)

    def _augment(self, img: np.ndarray, pts: np.ndarray):
        if not self.is_train:
            return img, pts

        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)

        angle = np.random.uniform(-self.rot_deg, self.rot_deg)
        scale = 1.0 + np.random.uniform(-self.scale_range, self.scale_range)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        img = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        pts = (pts_h @ M.T).astype(np.float32)

        if self.intensity_jitter > 0:
            alpha = 1.0 + np.random.uniform(-self.intensity_jitter, self.intensity_jitter)
            beta = np.random.uniform(-20.0, 20.0)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return img, pts

    def __getitem__(self, index):
        img_path, ann_path = self.samples[index]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pts = self._load_points(ann_path)
        original_pts = pts.copy()

        img, pts = self._augment(img, pts)

        orig_h, orig_w = img.shape[:2]
        out_w, out_h = self.image_size

        img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        sx = out_w / float(orig_w)
        sy = out_h / float(orig_h)

        pts_resized = pts.copy()
        pts_resized[:, 0] *= sx
        pts_resized[:, 1] *= sy

        hm_w, hm_h = self.heatmap_size
        target = np.zeros((self.num_joints, hm_h, hm_w), dtype=np.float32)
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)

        hm_x_ratio = hm_w / float(out_w)
        hm_y_ratio = hm_h / float(out_h)

        for j in range(self.num_joints):
            x, y = pts_resized[j]
            hx = x * hm_x_ratio
            hy = y * hm_y_ratio
            if hx < 0 or hy < 0 or hx >= hm_w or hy >= hm_h:
                target_weight[j, 0] = 0.0
                continue
            target[j] = draw_gaussian(target[j], np.array([hx, hy], dtype=np.float32), self.sigma)

        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        img_norm = img_norm.transpose(2, 0, 1)

        sample_id = img_path.stem
        pixel_size = self.pixel_size_map.get(sample_id, np.nan)

        meta = {
            "image_id": sample_id,
            "coords_original": torch.tensor(original_pts, dtype=torch.float32),
            "resize_factors": torch.tensor([sx, sy], dtype=torch.float32),
            "pixel_size_mm": torch.tensor(pixel_size, dtype=torch.float32),
        }

        return (
            torch.tensor(img_norm, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(target_weight, dtype=torch.float32),
            meta,
        )


def decode_heatmaps_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    b, j, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, j, -1)
    idx = flat.argmax(dim=-1)
    x = (idx % w).float()
    y = (idx // w).float()
    return torch.stack([x, y], dim=-1)


def weighted_heatmap_mse(
    pred: torch.Tensor, target: torch.Tensor, target_weight: torch.Tensor, criterion: nn.Module
) -> torch.Tensor:
    loss = criterion(pred, target).mean(dim=(2, 3))
    loss = loss * target_weight.squeeze(-1)
    return loss.sum() / target_weight.squeeze(-1).sum().clamp_min(1.0)


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, use_amp, grad_clip_norm):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets, target_weight, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_weight = target_weight.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device, enabled=use_amp and device == "cuda"):
            outputs = model(images)
            loss = weighted_heatmap_mse(outputs, targets, target_weight, criterion)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, criterion, image_size, heatmap_size):
    model.eval()
    running_loss = 0.0
    all_px_errors = []
    all_mm_errors = []

    hm_w, hm_h = heatmap_size
    img_w, img_h = image_size

    for images, targets, target_weight, meta in tqdm(loader, desc="Valid", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_weight = target_weight.to(device, non_blocking=True)

        outputs = model(images)
        loss = weighted_heatmap_mse(outputs, targets, target_weight, criterion)
        running_loss += loss.item()

        pred_hm = decode_heatmaps_argmax(outputs)
        pred_img = pred_hm.clone()
        pred_img[..., 0] *= img_w / float(hm_w)
        pred_img[..., 1] *= img_h / float(hm_h)

        resize_factors = meta["resize_factors"].to(device)
        gt_original = meta["coords_original"].to(device)

        pred_original = pred_img.clone()
        pred_original[..., 0] /= resize_factors[:, None, 0]
        pred_original[..., 1] /= resize_factors[:, None, 1]

        dists_px = torch.linalg.norm(pred_original - gt_original, dim=-1)
        all_px_errors.append(dists_px.cpu())

        pixel_size_mm = meta["pixel_size_mm"].to(device)
        valid_mm_mask = torch.isfinite(pixel_size_mm)
        if valid_mm_mask.any():
            dists_mm = dists_px[valid_mm_mask] * pixel_size_mm[valid_mm_mask][:, None]
            all_mm_errors.append(dists_mm.cpu())

    eval_loss = running_loss / max(len(loader), 1)
    mre_px = torch.cat(all_px_errors, dim=0).mean().item() if all_px_errors else float("nan")
    mre_mm = torch.cat(all_mm_errors, dim=0).mean().item() if all_mm_errors else float("nan")

    return eval_loss, mre_px, mre_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet for cephalometric landmark detection")
    parser.add_argument("--data-root", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "output" / "unet_ceph_landmark_detection")
    parser.add_argument("--annotator", type=str, default="Senior Orthodontists")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="valid")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--heatmap-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--base-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rot-deg", type=float, default=8.0)
    parser.add_argument("--scale-range", type=float, default=0.08)
    parser.add_argument("--intensity-jitter", type=float, default=0.10)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--unet-base-channels", type=int, default=32)
    parser.add_argument("--unet-backbone", type=str, default="convnext_base")
    parser.add_argument("--unet-in-channels", type=int, default=3)
    parser.add_argument(
        "--target-landmarks",
        type=str,
        nargs="+",
        default=[
            "S",
            "Go",
            "ANS",
            "B",
            "Me",
            "A",
            "UIT",
            "UIA",
            "LIT",
            "LIA",
            "Pn",
            "Sn",
            "Ls",
            "Li",
            "Pog",
            "Po",
            "Or",
            "N",
            "Pog`",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = torch.cuda.is_available()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    target_landmarks = list(dict.fromkeys(args.target_landmarks))
    num_joints = len(target_landmarks)

    train_ds = CephLandmarkDataset(
        data_root=args.data_root,
        split=args.train_split,
        annotator=args.annotator,
        target_landmarks=target_landmarks,
        image_size=tuple(args.image_size),
        heatmap_size=tuple(args.heatmap_size),
        sigma=args.sigma,
        num_joints=num_joints,
        is_train=True,
        rot_deg=args.rot_deg,
        scale_range=args.scale_range,
        intensity_jitter=args.intensity_jitter,
    )

    val_ds = CephLandmarkDataset(
        data_root=args.data_root,
        split=args.val_split,
        annotator=args.annotator,
        target_landmarks=target_landmarks,
        image_size=tuple(args.image_size),
        heatmap_size=tuple(args.heatmap_size),
        sigma=args.sigma,
        num_joints=num_joints,
        is_train=False,
    )

    test_ds = CephLandmarkDataset(
        data_root=args.data_root,
        split=args.test_split,
        annotator=args.annotator,
        target_landmarks=target_landmarks,
        image_size=tuple(args.image_size),
        heatmap_size=tuple(args.heatmap_size),
        sigma=args.sigma,
        num_joints=num_joints,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = build_model(
        model_name="unet",
        num_joints=num_joints,
        image_size=tuple(args.image_size),
        heatmap_size=tuple(args.heatmap_size),
        pretrained_path=None,
        resnet50_pretrained=False,
        unet_base_channels=args.unet_base_channels,
        unet_backbone=args.unet_backbone,
        unet_in_channels=args.unet_in_channels,
    ).to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device == "cuda")

    start_epoch = 1
    best_val_mre_mm = float("inf")
    best_val_mre_px = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mre_px": [], "val_mre_mm": []}

    if args.resume_checkpoint is not None and args.resume_checkpoint.exists():
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mre_mm = ckpt.get("best_val_mre_mm", best_val_mre_mm)
        best_val_mre_px = ckpt.get("best_val_mre_px", best_val_mre_px)
        print(f"Resumed from epoch {ckpt['epoch']}")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)} | Valid samples: {len(val_ds)} | Test samples: {len(test_ds)}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            criterion,
            args.use_amp,
            args.grad_clip_norm,
        )
        val_loss, val_mre_px, val_mre_mm = validate(
            model,
            val_loader,
            device,
            criterion,
            tuple(args.image_size),
            tuple(args.heatmap_size),
        )
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mre_px"].append(val_mre_px)
        history["val_mre_mm"].append(val_mre_mm)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
            f"val_mre_px={val_mre_px:.3f} | val_mre_mm={val_mre_mm:.3f} | "
            f"time={elapsed/60:.1f}m"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_mre_mm": best_val_mre_mm,
            "best_val_mre_px": best_val_mre_px,
            "config": {
                "image_size": tuple(args.image_size),
                "heatmap_size": tuple(args.heatmap_size),
                "sigma": args.sigma,
                "num_joints": num_joints,
                "annotator": args.annotator,
            },
            "history": history,
        }

        if epoch % args.save_every == 0:
            torch.save(checkpoint, output_dir / "last_checkpoint.pth")

        is_best = False
        if not np.isnan(val_mre_mm):
            if val_mre_mm < best_val_mre_mm:
                best_val_mre_mm = val_mre_mm
                is_best = True
        else:
            if val_mre_px < best_val_mre_px:
                best_val_mre_px = val_mre_px
                is_best = True

        if is_best:
            checkpoint["best_val_mre_mm"] = best_val_mre_mm
            checkpoint["best_val_mre_px"] = best_val_mre_px
            torch.save(checkpoint, output_dir / "best_model.pth")
            print(f"  New best model saved (val_mre_px={val_mre_px:.3f}, val_mre_mm={val_mre_mm:.3f})")

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    best_model_path = output_dir / "best_model.pth"
    if best_model_path.exists():
        best_ckpt = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(best_ckpt["model_state_dict"])
        print("Loaded best checkpoint for final test evaluation.")

    test_loss, test_mre_px, test_mre_mm = validate(
        model,
        test_loader,
        device,
        criterion,
        tuple(args.image_size),
        tuple(args.heatmap_size),
    )
    print(f"Test results | loss={test_loss:.5f} | mre_px={test_mre_px:.3f} | mre_mm={test_mre_mm:.3f}")

    test_metrics_df = pd.DataFrame(
        [
            {
                "split": args.test_split,
                "loss": test_loss,
                "mre_px": test_mre_px,
                "mre_mm": test_mre_mm,
            }
        ]
    )
    test_metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)

    print(f"Training complete. Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    main()
