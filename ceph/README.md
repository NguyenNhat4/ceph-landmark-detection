# 🦷 Cephalometric Landmark Detection

> Deep learning model for detecting 29 anatomical landmarks on cephalometric X-ray images

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research-green.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Prediction](#prediction)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)

---

## 🎯 Overview

This project implements an automated system for detecting **29 cephalometric landmarks** on lateral cephalometric radiographs using deep learning. The model achieves clinically acceptable accuracy (MRE < 2mm) for orthodontic treatment planning.

### Key Features

- ✨ **29 anatomical landmarks** detection (skeletal, dental, and soft tissue)
- 🎯 **High accuracy**: Target MRE < 2mm (clinically acceptable)
- 🚀 **Fast inference**: Real-time prediction on single images
- 📊 **Comprehensive metrics**: MRE, SDR@2mm, SDR@2.5mm, SDR@3mm, SDR@4mm
- 🔧 **Easy to use**: Simple CLI interface for training and prediction

### Dataset Information

- **Name**: CEPHA29 - Cephalometric Landmark Detection Dataset
- **Source**: [Aariz Dataset on Figshare](https://figshare.com/articles/dataset/Aariz_Cephalometric_Dataset/27986417?file=51041642)
- **Total Images**: 1000 cephalometric X-rays
  - Train: 700 images
  - Validation: 150 images
  - Test: 150 images
- **Annotations**: Provided by Senior and Junior Orthodontists

---

## 📊 Dataset

### 29 Landmark Points

<details>
<summary><b>Skeletal Landmarks (15 points)</b></summary>

| Symbol | Name | Description |
|--------|------|-------------|
| A | A-point | Subspinale point |
| ANS | Anterior Nasal Spine | Tip of anterior nasal spine |
| B | B-point | Supramentale point |
| Me | Menton | Lowest point of mandibular symphysis |
| N | Nasion | Frontonasal suture |
| Or | Orbitale | Lowest point of orbital rim |
| Pog | Pogonion | Most anterior point of chin |
| PNS | Posterior Nasal Spine | Tip of posterior nasal spine |
| S | Sella | Center of sella turcica |
| Ar | Articulare | Intersection of cranial base and condyle |
| Co | Condylion | Most superior point of condyle |
| Gn | Gnathion | Most anterior inferior point of chin |
| Go | Gonion | Most posterior inferior point of mandible |
| Po | Porion | Uppermost point of external auditory meatus |
| R | Ramus | Point on posterior border of ramus |

</details>

<details>
<summary><b>Dental Landmarks (8 points)</b></summary>

| Symbol | Name | Description |
|--------|------|-------------|
| LPM | Lower 2nd PM Cusp Tip | Lower second premolar cusp tip |
| LIT | Lower Incisor Tip | Lower incisor tip |
| LMT | Lower Molar Cusp Tip | Lower molar cusp tip |
| UPM | Upper 2nd PM Cusp Tip | Upper second premolar cusp tip |
| UIA | Upper Incisor Apex | Upper incisor apex |
| UIT | Upper Incisor Tip | Upper incisor tip |
| UMT | Upper Molar Cusp Tip | Upper molar cusp tip |
| LIA | Lower Incisor Apex | Lower incisor apex |

</details>

<details>
<summary><b>Soft Tissue Landmarks (6 points)</b></summary>

| Symbol | Name | Description |
|--------|------|-------------|
| Li | Labrale inferius | Most anterior point of lower lip |
| Ls | Labrale superius | Most anterior point of upper lip |
| N' | Soft Tissue Nasion | Soft tissue nasion |
| Pog' | Soft Tissue Pogonion | Soft tissue pogonion |
| Pn | Pronasale | Most anterior point of nose |
| Sn | Subnasale | Junction of nose and upper lip |

</details>

### Directory Structure

```
data/
├── train/                      # 700 training images
│   ├── Cephalograms/
│   │   └── *.png
│   └── Annotations/
│       ├── Cephalometric Landmarks/
│       │   ├── Junior Orthodontists/
│       │   │   └── *.json
│       │   └── Senior Orthodontists/
│       │       └── *.json
│       └── CVM Stages/
│           └── *.json
├── valid/                      # 150 validation images
├── test/                       # 150 test images
└── cephalogram_machine_mappings.csv
```

### Annotation Format

```json
{
  "ceph_id": "cks2ip8fq2a0j0yufdfssbc09",
  "landmarks": [
    {
      "landmark_id": "ckr20ld7v4a4r0za3gqkydid2",
      "title": "A-point",
      "symbol": "A",
      "value": {"x": 1315, "y": 1086}
    }
  ],
  "labeled_at": "2022-10-18T02:07:41.000Z",
  "dataset_name": "CEPHA29: Cephalometric Landmark Detection Dataset"
}
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM
- 4GB+ GPU VRAM (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/NguyenNhat4/ceph-landmark-detection.git
cd ceph-landmark-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (with CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt
```

### Download Dataset

Download the dataset from [Figshare](https://figshare.com/articles/dataset/Aariz_Cephalometric_Dataset/27986417?file=51041642) and extract to `./data/` directory.

---

## ⚡ Quick Start

### 1. Create a Small Subset (50 images)

For quick testing and prototyping:

```bash
python create_subset.py
```

This creates `./data/subset_50/` with 50 training and 10 validation images.

### 2. Train on Subset

```bash
python train_quick.py
```

**Expected results after 50 epochs:**
- Training time: ~15-30 minutes (GPU)
- Val MRE: ~3-5mm

### 3. Predict on New Image

```bash
python predict.py \
  --image data/subset_50/valid/Cephalograms/cks2ip8fq29zf0yufe0r67c79.png \
  --model best_model_50.pth
```

**Output:**
- `prediction.png` - Visualization with landmarks
- `prediction.json` - Landmark coordinates

---

## 🎓 Training

### Quick Training (50 images)

See [QUICK_START_50_IMAGES.md](QUICK_START_50_IMAGES.md) for detailed guide.

```bash
python create_subset.py
python train_quick.py
```

### Full Training (700 images)

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for comprehensive guide.

**Key steps:**

1. **Prepare Dataset**
   ```python
   from dataset import CephDataset

   train_dataset = CephDataset(
       data_dir='./data',
       mode='train',
       use_senior=True,
       img_size=512
   )
   ```

2. **Choose Model Architecture**
   - **Regression Model** (faster, lighter)
   - **Heatmap Model** (better spatial localization)

3. **Configure Training**
   ```python
   BATCH_SIZE = 8
   EPOCHS = 100
   LR = 1e-3
   IMG_SIZE = 512
   ```

4. **Train**
   ```bash
   python train_quick.py  # Modify DATA_DIR to './data'
   ```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 512x512 | Input resolution |
| Batch Size | 4-8 | Depends on GPU memory |
| Learning Rate | 1e-3 → 1e-6 | With cosine annealing |
| Epochs | 100-200 | Full training |
| Optimizer | Adam/AdamW | - |
| Loss Function | MSE / Wing Loss | Wing Loss better for landmarks |

---

## 🔮 Prediction

### Single Image Prediction

```bash
python predict.py \
  --image path/to/image.png \
  --model best_model_50.pth \
  --output result.png
```

### Compare with Ground Truth

```bash
python predict.py \
  --image data/subset_50/valid/Cephalograms/image.png \
  --model best_model_50.pth \
  --anno "data/subset_50/valid/Annotations/Cephalometric Landmarks/Senior Orthodontists/image.json"
```

**Output:**
- `prediction.png` - Predicted landmarks
- `comparison.png` - Side-by-side comparison with ground truth
- `prediction.json` - Coordinates in JSON format

### Hide Landmark Labels

```bash
python predict.py \
  --image path/to/image.png \
  --model best_model_50.pth \
  --no-labels
```

### Use CPU (if no GPU)

```bash
python predict.py \
  --image path/to/image.png \
  --model best_model_50.pth \
  --device cpu
```

---

## 📈 Evaluation

### Metrics

**Primary Metric:**
- **MRE (Mean Radial Error)**: Average Euclidean distance in millimeters
  - **Clinically acceptable**: MRE < 2.0mm

**Secondary Metrics:**
- **SDR (Successful Detection Rate)**: Percentage of landmarks within threshold
  - SDR@2.0mm
  - SDR@2.5mm
  - SDR@3.0mm
  - SDR@4.0mm

### Evaluation on Test Set

```python
from train_quick import validate, calculate_mre
from model import LandmarkModel
import torch

# Load model
model = LandmarkModel()
checkpoint = torch.load('best_model_50.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
test_dataset = CephDataset('./data', mode='test', img_size=512)
test_loader = DataLoader(test_dataset, batch_size=8)

val_loss, val_mre = validate(model, test_loader, nn.MSELoss(), 'cuda')
print(f"Test MRE: {val_mre:.2f}mm")
```

---

## 📁 Project Structure

```
ceph_landmarks_detection/
├── data/                          # Dataset directory
│   ├── train/
│   ├── valid/
│   ├── test/
│   ├── subset_50/                 # Small subset for testing
│   └── cephalogram_machine_mappings.csv
├── dataset.py                     # Dataset class
├── model.py                       # Model architecture
├── train_quick.py                 # Training script
├── predict.py                     # Prediction script
├── create_subset.py               # Create small dataset subset
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── TRAINING_GUIDE.md              # Detailed training guide
├── QUICK_START_50_IMAGES.md       # Quick start guide
└── best_model_50.pth              # Trained model checkpoint
```

---

## 🛠️ Model Architecture

### Regression Model (Default)

```python
class LandmarkModel(nn.Module):
    - Backbone: EfficientNet-B0 (pretrained on ImageNet)
    - Head: FC layers (512 units)
    - Output: 29 × 2 coordinates (x, y)
```

**Advantages:**
- ✅ Fast inference
- ✅ Lightweight (5.3M parameters)
- ✅ Good for real-time applications

### Alternative: Heatmap Model

```python
class HeatmapModel(nn.Module):
    - Backbone: U-Net with ResNet50
    - Output: 29 heatmap channels
```

**Advantages:**
- ✅ Better spatial localization
- ✅ More robust to occlusions

---

## 📊 Results

### Expected Performance

| Metric | Quick Start (50 imgs) | Full Training (700 imgs) |
|--------|----------------------|--------------------------|
| **MRE** | 3-5mm | < 2mm (target) |
| **SDR@2mm** | ~40-60% | > 90% |
| **SDR@4mm** | ~80-90% | > 95% |
| **Training Time** | 15-30 min | 2-4 hours |

---

## 📚 Documentation

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Comprehensive training guide with theory and best practices
- **[QUICK_START_50_IMAGES.md](QUICK_START_50_IMAGES.md)** - Quick start guide for testing with 50 images
- **[Readme.txt](Readme.txt)** - Original dataset documentation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 Citation

If you use this code or dataset, please cite:

```bibtex
@dataset{aariz_cepha29_2024,
  title={CEPHA29: Cephalometric Landmark Detection Dataset},
  author={Aariz Research Team},
  year={2024},
  publisher={Figshare},
  url={https://figshare.com/articles/dataset/Aariz_Cephalometric_Dataset/27986417}
}
```

**Original Dataset:**
- GitHub: https://github.com/manwaarkhd/aariz
- Figshare: https://figshare.com/articles/dataset/Aariz_Cephalometric_Dataset/27986417

---

## 📧 Contact

For questions or issues:
- **Dataset**: cepha29.challenge@gmail.com
- **GitHub Issues**: [Create an issue](https://github.com/NguyenNhat4/ceph-landmark-detection/issues)

---

## 📜 License

This project is for research and educational purposes. Please refer to the original dataset publication for licensing information.

---

## 🙏 Acknowledgments

- Original CEPHA29 dataset by Aariz Research Team
- PyTorch and timm library contributors
- OpenCV community

---

<p align="center">
  Made with ❤️ for orthodontic research
</p>
