import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrnet import HighResolutionNet

try:
    import torchvision
    from torchvision.models import ResNet50_Weights
except Exception:
    torchvision = None
    ResNet50_Weights = None

try:
    from backbones_unet.model.unet import Unet as PretrainedUnet
except Exception:
    PretrainedUnet = None


HRNET_W32_EXTRA = {
    "FINAL_CONV_KERNEL": 1,
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [32, 64],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
    "STAGE3": {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [32, 64, 128],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
    "STAGE4": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [32, 64, 128, 256],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
}


class AttrDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        value = self[key]
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def build_hrnet_config(num_joints: int, pretrained_path: Optional[str] = None) -> AttrDict:
    cfg = AttrDict()
    cfg.MODEL = AttrDict()
    cfg.MODEL.NUM_JOINTS = num_joints
    cfg.MODEL.EXTRA = AttrDict(copy.deepcopy(HRNET_W32_EXTRA))
    cfg.MODEL.PRETRAINED = pretrained_path or ""
    cfg.MODEL.INIT_WEIGHTS = False
    return cfg


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_hrnet_pretrained(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = safe_torch_load(checkpoint_path)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break

    clean_state = {}
    for key, value in state_dict.items():
        clean_state[key.replace("module.", "")] = value

    model_state = model.state_dict()
    matched = {
        key: value
        for key, value in clean_state.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing = len(model_state) - len(matched)

    model_state.update(matched)
    model.load_state_dict(model_state)

    print(f"Loaded pretrained params: {len(matched)} | Missing in pretrained: {missing}")


def load_checkpoint_state(model: nn.Module, checkpoint_path: str, strict: bool = False):
    checkpoint = safe_torch_load(checkpoint_path)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    clean_state = {key.replace("module.", ""): value for key, value in state_dict.items()}
    return model.load_state_dict(clean_state, strict=strict)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetHeatmap(nn.Module):
    def __init__(self, num_joints: int, heatmap_size: Tuple[int, int], base_channels: int = 32):
        super().__init__()
        self.heatmap_size = heatmap_size

        self.enc1 = DoubleConv(3, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(2)

        self.dec3 = DoubleConv(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = DoubleConv(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = DoubleConv(base_channels * 2 + base_channels, base_channels)

        self.head = nn.Conv2d(base_channels, num_joints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        up3 = F.interpolate(enc4, size=enc3.shape[-2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))

        up2 = F.interpolate(dec3, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = F.interpolate(dec2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        logits = self.head(dec1)
        return F.interpolate(
            logits,
            size=(self.heatmap_size[1], self.heatmap_size[0]),
            mode="bilinear",
            align_corners=False,
        )


class PretrainedUNetHeatmap(nn.Module):
    def __init__(
        self,
        num_joints: int,
        heatmap_size: Tuple[int, int],
        backbone: str = "convnext_base",
        in_channels: int = 3,
    ):
        super().__init__()
        if PretrainedUnet is None:
            raise ImportError("backbones-unet is required for pretrained UNet models")

        self.heatmap_size = heatmap_size
        self.model = PretrainedUnet(
            backbone=backbone,
            in_channels=in_channels,
            num_classes=num_joints,
        )
        print(f"Using pretrained UNet backbone: {backbone}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return F.interpolate(
            logits,
            size=(self.heatmap_size[1], self.heatmap_size[0]),
            mode="bilinear",
            align_corners=False,
        )


class ResNet50Heatmap(nn.Module):
    def __init__(self, num_joints: int, heatmap_size: Tuple[int, int], pretrained: bool = True):
        super().__init__()
        if torchvision is None:
            raise ImportError("torchvision is required for resnet50 models")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = torchvision.models.resnet50(weights=weights)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_joints, kernel_size=1),
        )
        self.heatmap_size = heatmap_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return F.interpolate(
            logits,
            size=(self.heatmap_size[1], self.heatmap_size[0]),
            mode="bilinear",
            align_corners=False,
        )


def build_model(
    model_name: str,
    num_joints: int,
    image_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    pretrained_path: Optional[str] = None,
    resnet50_pretrained: bool = True,
    unet_base_channels: int = 32,
    unet_backbone: str = "convnext_base",
    unet_in_channels: int = 3,
) -> nn.Module:
    name = model_name.lower().strip()

    if name in {"hrnet", "hrnet-w32", "hrnet_w32"}:
        cfg = build_hrnet_config(num_joints=num_joints, pretrained_path=pretrained_path)
        model = HighResolutionNet(cfg)
        if pretrained_path:
            load_hrnet_pretrained(model, pretrained_path)
        return model

    if name in {"unet", "u-net"}:
        if PretrainedUnet is not None:
            return PretrainedUNetHeatmap(
                num_joints=num_joints,
                heatmap_size=heatmap_size,
                backbone=unet_backbone,
                in_channels=unet_in_channels,
            )

        return UNetHeatmap(
            num_joints=num_joints,
            heatmap_size=heatmap_size,
            base_channels=unet_base_channels,
        )

    if name in {"resnet50", "resnet-50"}:
        return ResNet50Heatmap(
            num_joints=num_joints,
            heatmap_size=heatmap_size,
            pretrained=resnet50_pretrained,
        )

    raise ValueError("Unsupported model_name. Use 'hrnet', 'unet', or 'resnet50'.")
