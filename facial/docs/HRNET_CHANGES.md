# Understanding HRNet for Facial Landmark Detection: Architecture Fixes

This document explains the critical architectural fixes made to the `train.ipynb` notebook to properly align it with the official [HRNet-Facial-Landmark-Detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection) repository.

## 1. The `timm` Features Extraction Issue

**What was wrong:**
Initially, the code loaded the HRNet backbone using `timm.create_model('hrnet_w18', features_only=True)`. 
While `features_only=True` works perfectly for models like ResNet or EfficientNet by returning the intermediate feature maps after each pooling block, it fundamentally breaks the design philosophy of HRNet. 

If you look at the `timm` output shapes for `features_only=True`:
```python
Feature 0 shape: torch.Size([1, 64, 256, 256])  # Stride 2
Feature 1 shape: torch.Size([1, 128, 128, 128]) # Stride 4
Feature 2 shape: torch.Size([1, 256, 64, 64])   # Stride 8
Feature 3 shape: torch.Size([1, 512, 32, 32])   # Stride 16
```
They represent heavily downsampled, ResNet-style representations pulled from the Classification Head. HRNet’s superpower is maintaining a high-resolution branch throughout the entire network, not destroying it through sequential pooling.

**The Fix:**
We loaded the full model (`pretrained=False` without `features_only`) and manually intercepted the outputs at the end of the parallel HR stages:
```python
# Returns a List of the 4 parallel, multi-resolution branches!
hr_features = self.backbone.stages(x) 
```

## 2. Facial Landmark Aggregation Head

**What was wrong:**
The previous code simply took the highest-resolution map (which we mistakenly assumed was `features[0]`) and directly applied a $1\times1$ convolution to predict the keypoints. Because landmarks only used small local receptive fields, the model failed to understand the global structure of the face (e.g., distinguishing between a left eye and right eye).

**The Fix:**
I inspected the official HRNet Facial Landmark `hrnet.py` file. The original authors specifically design their prediction head to **combine all 4 resolutions**.

We implemented the explicit feature aggregation logic:
1. Extract the 4 branches, which have channel sizes `[18, 36, 72, 144]`.
2. Upsample the smaller feature maps (`36`, `72`, `144`) back to the size of the highest resolution branch (`18`).
3. Concatenate them together along the channel dimension to form a dense representation of $18 + 36 + 72 + 144 = 270$ channels.
4. Pass this concatenated $270$-channel feature map through a `Sequential` block consisting of a $1\times1$ convolution $\rightarrow$ Batch Normalization $\rightarrow$ ReLU $\rightarrow$ final $1\times1$ linear projection into `num_keypoints`.

```python
h, w = hr_features[0].shape[2:]
out = [hr_features[0]]
for i in range(1, len(hr_features)):
    out.append(F.interpolate(hr_features[i], size=(h, w), mode='bilinear', align_corners=False))
    
out_cat = torch.cat(out, dim=1) # Shape: (B, 270, 128, 128)
heatmaps = self.final_head(out_cat)
```

## 3. Resolving Spatial Strides

**What was wrong:**
There was massive confusion between the Ground Truth Heatmap sizes and the predicted tensor sizes from PyTorch.
- Due to `features_only=True`, the notebook was expecting a `256x256` output (Stride 2).
- True HRNet parallel branches run at **Stride 4** relative to the input image.

**The Fix:**
Because the true high-resolution stage output runs at Stride 4, an input image of `512x512` will permanently and invariably produce a highest-resolution feature map of `128x128`.

I corrected the Custom Dataset so the `heatmap_size` matches the Stride 4 backbone output:
```python
# Input Image: 512x512
# Output Heatmap (Stride 4): 128x128
dataset = CephalometricDataset(..., image_size=(512, 512), heatmap_size=(128, 128))
```

And symmetrically adjusted the post-processing inference script to scale the `128` coordinate grid back up to `512` by using a multiplier of `4.0`:
```python
preds_x *= 4.0
preds_y *= 4.0
```

### Summary of what you learned:
When using models like HRNet, the default wrapper features in libraries like `timm` are often tailored for Image Classification. If you want to use them for dense pixel-prediction tasks (pose, landmarks, segmentation), you almost always have to bypass the classification classification head and manually aggregate the raw multi-scale internal stages identically to the original academic paper.