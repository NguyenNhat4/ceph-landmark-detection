# Cephalometric Data Augmentation Strategy

This document serves as a reference for how data augmentation is applied to the small cephalometric landmark dataset, directly mirroring the techniques used in the official HRNet codebase but tailored for lateral (side-profile) medical images.

## Overview
Because the dataset is small, data augmentation is crucial to prevent overfitting. We apply three standard pose-estimation augmentations: **Scale**, **Rotation**, and **Horizontal Flipping**. 

These augmentations are applied *before* the final affine crop and target heatmap generation.

## 1. Scale Augmentation
* **Hyperparameter:** `scale_factor = 0.25`
* **Mechanism:** The bounding box scale is randomly multiplied by a factor between `0.75` (zoomed out) and `1.25` (zoomed in).
* **Code:** `scale = scale * random.uniform(1 - self.scale_factor, 1 + self.scale_factor)`

## 2. Rotation Augmentation
* **Hyperparameter:** `rot_factor = 30`
* **Mechanism:** There is a 60% probability that the image will be rotated. If hit, a random angle between `-30` and `+30` degrees is selected.
* **How it applies natively:** The selected rotation `r` is passed directly into the HRNet `crop_v2` and `transform_pixel` functions. HRNet's affine matrix math handles spinning the image matrix and recalculating the ground-truth Gaussian coordinates simultaneously.

## 3. Horizontal Flipping (The "Side-Profile" Difference)
* **Hyperparameter:** `flip = True` (50% probability)
* **The Problem with Standard Faces:** In standard datasets like 300W or WFLW, flipping an image horizontally means the "Left Eye" (e.g., index 36) physically becomes the "Right Eye" (index 45). The arrays require a complex `fliplr_joints` matching algorithm to swap indices.
* **The Cephalometric Solution:** Lateral cephalograms are 2D side-profiles. The landmarks (Glabella, Pronasal, Subnasale, etc.) rest primarily on the midline profile of the face. Therefore, they do not have left/right symmetrical counterparts to swap. 
* **Mechanism:** To flip a skull horizontally, we only need to mirror the `X` coordinate against the image width, keeping the label indices exactly the same.
* **Formula:** `new_x = img_width - original_x - 1`

## Implementation Reference
This is the core logic snippet placed inside the custom `CephalometricDataset`'s `__getitem__` method:

```python
r = 0
if self.is_train:
    # 1. Random Scale
    scale = scale * random.uniform(1 - self.scale_factor, 1 + self.scale_factor)
    
    # 2. Random Rotation (60% chance)
    if random.random() <= 0.6:
        r = random.uniform(-self.rot_factor, self.rot_factor)
        
    # 3. Random Horizontal Flip (50% chance)
    if random.random() <= 0.5 and self.flip:
        image = cv2.flip(image, 1) # Flip image horizontally
        img_width = image.shape[1]
        
        # Mirror the center of the bounding box
        center[0] = img_width - center[0] - 1
        
        # Mirror the X coordinate for every visible landmark
        for i in range(pts.shape[0]):
            if pts[i, 2] > 0:
                pts[i, 0] = img_width - pts[i, 0] - 1

# Apply transformations
img_crop = crop_v2(image, center, scale, self.image_size, rot=r)

# ... inside the loop generating target heatmaps ...
tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, center, scale, self.heatmap_size, rot=r)
```