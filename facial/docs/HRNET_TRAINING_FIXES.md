# HRNet Training: Key Structural Fixes
*Date: March 2026*

This document breaks down two major structural flaws that were causing the HRNet model to fail during training and hallucinate badly during inference.

---

## 1. The Bounding Box Mismatch (Train vs. Inference Domain Shift)

**The Problem:**
In the original `CephalometricDataset`, the bounding box used for cropping was derived directly from the COCO annotations (`ann['bbox']`). For facial keypoints, this bounding box was wrapped very tightly around the keypoints themselves (lips, nose, chin). As a result, the training images were scaled-up, heavily zoomed-in crops of the facial profile surface.

However, during **inference**, without a ground-truth bounding box, the script assumed the bounding box was the *entire image*:
```python
orig_h, orig_w = image_rgb.shape[:2]
center_w = orig_w / 2.0
scale = max(orig_w, orig_h) / 200.0 * 1.25 # Center on the whole cranium
```
When HRNet was fed this zoomed-out full scale image during inference, it failed completely because it had exclusively been trained on tight facial crops.

**The Fix:**
Updated `CephalometricDataset` (Cell 3) to use the entire image as the bounding box, precisely matching the math used in the inference script:
```python
orig_w = img_info['width']
orig_h = img_info['height']
center_w = orig_w / 2.0
center_h = orig_h / 2.0
center = np.array([center_w, center_h], dtype=np.float32)
scale = max(orig_w, orig_h) / 200.0 * 1.25
```
*Note: Because medical Cephalograms are highly standardized in their layout, training the network on the whole global image context ensures that your training domain exactly matches your inference domain.*

---

## 2. Loss Scaling & Weight Decay Collapse

**The Problem:**
The `WeightedMSELoss` calculated the spatial error using `.mean(dim=1)`:
```python
loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt).mean(dim=1) * weight.squeeze(-1)
```
A 128x128 target heatmap contains 16,384 pixels, but only a tiny region around the 2D Gaussian peak has non-zero values. Taking the mean divided the target gradients by $16384$, resulting in phenomenally small gradient signals (e.g., `~0.00003`).

Compounding this issue, the code used `AdamW` combined with a high learning rate (`1e-3`). `AdamW` isolates weight decay directly onto the model's weights (default `decay=0.01`). Because the gradient signal from the MSE loss was physically crushed by the spatial mean, the `0.01` weight decay massively overpowered the data gradient. This forced the model's weights to constantly collapse towards zero, completely stalling training.

**The Fix:**
1. **Sum instead of Mean:** Replaced `.mean(dim=1)` with `.sum(dim=1)` inside the spatial loss calculation so the sparse Gaussian target gradients aren't diluted.
2. **Removed Hacky Multipliers:** Removed the `loss = ... * 1000` patch in the training loop now that the loss has a healthy numerical scale.
3. **Optimized Optimizer:** Switched `AdamW` to standard `Adam` to prevent decoupled weight decay from stalling the network gradients.
4. **Tuned LR:** Lowered the learning rate from `1e-3` to `1e-4`, which is considered the standard stable size for fine-tuning pre-trained HRNet/ImageNet features.
