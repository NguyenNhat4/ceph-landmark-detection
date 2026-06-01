import os
from pathlib import Path

# Dynamically resolve path relative to the app directory
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "ceph" / "models" / "ceph_hrnet_notebook"

# Model configuration
MODEL_PATH = MODELS_DIR / "best_model.pth"
NUM_JOINTS = 29
IMAGE_SIZE = (512, 512)
HEATMAP_SIZE = (128, 128)
USE_AMP = True

# Landmark symbol mapping (index to symbol) - must match training data order
LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]

# Mapping to match the paper's symbols
PAPER_MAPPING = {
    "Ls": "ls",
    "Li": "li",
    "Pog`": "Pg'",
    "Pog": "Pg",
    "Go": "go",
    "UIT": "I",
    "LIT": "i"
}

ALLOWED_LANDMARKS = {
    "S", "ANS", "B", "Me", "N", "A", "B", "go", "Me",
    "I", "UIA", "i", "LIA",
    "Pn", "Sn", "ls", "li", "Pg'", "Po", "Or", "N`", "Pg"
}

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
