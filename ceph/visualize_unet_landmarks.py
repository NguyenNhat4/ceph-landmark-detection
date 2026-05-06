import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from model_factory import build_model, load_checkpoint_state


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
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.annotator = annotator
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.target_landmarks = target_landmarks
        self.num_joints = len(target_landmarks)

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

    def __getitem__(self, index):
        img_path, ann_path = self.samples[index]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pts = self._load_points(ann_path)

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

        hm_x_ratio = hm_w / float(out_w)
        hm_y_ratio = hm_h / float(out_h)

        for j in range(self.num_joints):
            x, y = pts_resized[j]
            hx = x * hm_x_ratio
            hy = y * hm_y_ratio
            target[j] = draw_gaussian(target[j], np.array([hx, hy], dtype=np.float32), self.sigma)

        img_norm = img_resized.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std
        img_norm = img_norm.transpose(2, 0, 1)

        return torch.tensor(img_norm, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), img_path.stem


def decode_heatmaps_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    b, j, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, j, -1)
    idx = flat.argmax(dim=-1)
    x = (idx % w).float()
    y = (idx // w).float()
    return torch.stack([x, y], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize UNet landmark predictions")
    parser.add_argument("--data-root", type=Path, default=Path.cwd() / "data")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--annotator", type=str, default="Senior Orthodontists")
    parser.add_argument("--checkpoint", type=Path, default=Path.cwd() / "output" / "unet_ceph_landmark_detection" / "best_model.pth")
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path.cwd() / "output" / "unet_ceph_landmark_detection" / "visualizations",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--heatmap-size", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--num-samples", type=int, default=2)
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
            "B",
            "Me",
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    target_landmarks = list(dict.fromkeys(args.target_landmarks))
    num_joints = len(target_landmarks)

    dataset = CephLandmarkDataset(
        data_root=args.data_root,
        split=args.split,
        annotator=args.annotator,
        target_landmarks=target_landmarks,
        image_size=tuple(args.image_size),
        heatmap_size=tuple(args.heatmap_size),
        sigma=args.sigma,
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

    if args.checkpoint.exists():
        missing_keys, unexpected_keys = load_checkpoint_state(model, str(args.checkpoint), strict=False)
        if missing_keys or unexpected_keys:
            print(
                f"Checkpoint load report | missing={len(missing_keys)} | unexpected={len(unexpected_keys)}"
            )
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model.eval()

    if args.image_path is not None:
        image_path = args.image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        sample_idxs = [None]
    else:
        sample_idxs = random.sample(range(len(dataset)), k=min(args.num_samples, len(dataset)))
    hm_w, hm_h = args.heatmap_size
    img_w, img_h = args.image_size

    for idx in sample_idxs:
        if idx is None:
            image_id = image_path.stem
            ann_path = dataset.ann_dir / f"{image_id}.json"
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotation not found: {ann_path}")
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if bgr is None:
                raise ValueError(f"Failed to read image: {image_path}")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pts = dataset._load_points(ann_path)

            orig_h, orig_w = img.shape[:2]
            out_w, out_h = args.image_size
            img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            sx = out_w / float(orig_w)
            sy = out_h / float(orig_h)
            pts_resized = pts.copy()
            pts_resized[:, 0] *= sx
            pts_resized[:, 1] *= sy

            target = np.zeros((num_joints, hm_h, hm_w), dtype=np.float32)
            hm_x_ratio = hm_w / float(out_w)
            hm_y_ratio = hm_h / float(out_h)
            for j in range(num_joints):
                x, y = pts_resized[j]
                hx = x * hm_x_ratio
                hy = y * hm_y_ratio
                target[j] = draw_gaussian(target[j], np.array([hx, hy], dtype=np.float32), args.sigma)

            img_norm = img_resized.astype(np.float32) / 255.0
            img_norm = (img_norm - dataset.mean) / dataset.std
            img_norm = img_norm.transpose(2, 0, 1)

            image = torch.tensor(img_norm, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
        else:
            image, target, image_id = dataset[idx]
        pred_hm = model(image.unsqueeze(0).to(device))
        pred_xy = decode_heatmaps_argmax(pred_hm).squeeze(0).cpu().numpy()
        pred_xy[:, 0] *= img_w / float(hm_w)
        pred_xy[:, 1] *= img_h / float(hm_h)

        vis = image.permute(1, 2, 0).cpu().numpy()
        vis = (vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        vis = np.clip(vis, 0, 1)

        gt = target.numpy()
        gt_idx = np.argmax(gt.reshape(num_joints, -1), axis=1)
        gt_xy = np.stack([gt_idx % hm_w, gt_idx // hm_w], axis=1).astype(np.float32)
        gt_xy[:, 0] *= img_w / float(hm_w)
        gt_xy[:, 1] *= img_h / float(hm_h)

        plt.figure(figsize=(7, 7))
        plt.imshow(vis)
        plt.scatter(gt_xy[:, 0], gt_xy[:, 1], s=18, c="lime", label="GT")
        plt.scatter(pred_xy[:, 0], pred_xy[:, 1], s=18, c="red", label="Pred")
        plt.title(f"Sample: {image_id}")
        plt.legend()
        plt.axis("off")
        save_path = save_dir / f"{image_id}_unet_overlay.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        if not args.no_show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()
