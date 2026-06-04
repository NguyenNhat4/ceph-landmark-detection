import os
import sys
import contextlib
import copy
import cv2
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.config import (
    MODEL_PATH, NUM_JOINTS, IMAGE_SIZE, HEATMAP_SIZE, USE_AMP,
    LANDMARK_SYMBOLS, PAPER_MAPPING, ALLOWED_LANDMARKS, HRNET_W32_EXTRA
)

from .hrnet import HighResolutionNet

logger = logging.getLogger(__name__)

# Global state
MODEL = None
MODEL_LOADED = False


class AttrDict(dict):
    """Dictionary with attribute-style access"""
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


def build_hrnet_config(num_joints: int) -> AttrDict:
    """Build HRNet configuration"""
    cfg = AttrDict()
    cfg.MODEL = AttrDict()
    cfg.MODEL.NUM_JOINTS = num_joints
    cfg.MODEL.EXTRA = AttrDict(copy.deepcopy(HRNET_W32_EXTRA))
    cfg.MODEL.PRETRAINED = ''
    cfg.MODEL.INIT_WEIGHTS = False
    return cfg


def safe_torch_load(path: str):
    """Safely load torch checkpoint"""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def extract_state_dict(checkpoint_obj):
    """Extract state dict from checkpoint"""
    if isinstance(checkpoint_obj, torch.nn.Module):
        return checkpoint_obj.state_dict()

    if isinstance(checkpoint_obj, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
        return checkpoint_obj

    raise RuntimeError('Unsupported checkpoint format.')


def decode_heatmaps_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Decode heatmaps to coordinates using argmax"""
    b, j, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, j, -1)
    idx = flat.argmax(dim=-1)
    x = (idx % w).float()
    y = (idx // w).float()
    return torch.stack([x, y], dim=-1)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image from bytes"""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img_bgr is None:
        raise ValueError("Uploaded file is not a valid image.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_model():
    """Load the HRNet model"""
    global MODEL, MODEL_LOADED
    logger.info("Loading HRNet model...")
    
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model checkpoint not found at {MODEL_PATH}")
        return

    try:
        cfg = build_hrnet_config(num_joints=NUM_JOINTS)
        MODEL = HighResolutionNet(cfg)

        loaded_obj = safe_torch_load(str(MODEL_PATH))
        state_dict = extract_state_dict(loaded_obj)
        clean_state_dict = {
            (k[7:] if k.startswith('module.') else k): v
            for k, v in state_dict.items()
        }

        missing_keys, unexpected_keys = MODEL.load_state_dict(clean_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            logger.info(f"Checkpoint load report | missing: {len(missing_keys)} | unexpected: {len(unexpected_keys)}")

        MODEL.eval()

        if torch.cuda.is_available():
            MODEL.cuda()

        MODEL_LOADED = True
        logger.info("Model successfully loaded and ready for inference!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


def preprocess_image(img):
    """Preprocess image for model inference"""
    if img is None:
        raise ValueError("Could not read image for preprocessing")

    h, w, _ = img.shape
    out_w, out_h = IMAGE_SIZE
    img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize with ImageNet stats
    img_resized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_resized = (img_resized - mean) / std
    
    # Convert to PyTorch Tensor [Channels, Height, Width]
    img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]

    resize_factors = np.array([out_w / float(w), out_h / float(h)], dtype=np.float32)
    return img_tensor, w, h, resize_factors


def predict_landmarks(image: np.ndarray):
    """Run inference on image and extract landmarks"""
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded.")

    img_tensor, original_w, original_h, resize_factors = preprocess_image(image)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.inference_mode():
        if img_tensor.is_cuda and hasattr(torch, 'autocast') and USE_AMP:
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            outputs = MODEL(img_tensor)

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[-1]

    preds_hm = decode_heatmaps_argmax(outputs)
    preds = preds_hm.clone()
    preds[..., 0] *= IMAGE_SIZE[0] / float(HEATMAP_SIZE[0])
    preds[..., 1] *= IMAGE_SIZE[1] / float(HEATMAP_SIZE[1])

    preds[..., 0] /= resize_factors[0]
    preds[..., 1] /= resize_factors[1]

    confidence = outputs.detach().reshape(outputs.shape[0], outputs.shape[1], -1).max(dim=-1).values

    predicted_points = preds[0]
    confidence = confidence[0]

    # Return only allowed landmarks
    landmarks = []
    for i in range(len(predicted_points)):
        original_symbol = LANDMARK_SYMBOLS[i] if i < len(LANDMARK_SYMBOLS) else f"L{i+1}"
        symbol = PAPER_MAPPING.get(original_symbol, original_symbol)
        
        if symbol in ALLOWED_LANDMARKS:
            landmarks.append({
                "symbol": symbol,
                "original_symbol": original_symbol,
                "value": {
                    "x": float(predicted_points[i][0]),
                    "y": float(predicted_points[i][1])
                },
                "confidence": float(confidence[i])
            })      

    return landmarks, int(original_w), int(original_h), None
