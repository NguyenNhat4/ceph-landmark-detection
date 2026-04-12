# Inference Result Issue Explained

## Why were the inference results initially so bad compared to `train.ipynb`?

If you ever wonder why your initial test in `inference.ipynb` mapped all keypoints to random garbage locations while `train.ipynb` evaluated perfectly, here is exactly why it happened and how it was fixed.

### 1. The Architecture Mismatch
During training (`train.ipynb`), the model was defined using a custom Python class called `PoseHRNet`. 
This class did two things:
1. It used `timm.create_model('hrnet_w18')` as the backbone feature extractor.
2. It defined a custom `self.final_head` (a multi-resolution concatenation followed by some Conv2D layers) to predict exactly 8 keypoints across the concatenated scales.

When building `inference.ipynb`, we initially used the standard official HRNet loading method:
```python
model = models.get_face_alignment_net(config)
```
This loaded the official HRNet structure logic, which uses `model.final_layer` instead of `model.final_head`, and calculates features differently without `timm`'s naming structures.

### 2. The Weight Loading Trap (`strict=False`)
Because the official `get_face_alignment_net(config)` did not expect the `'backbone.'` dictionary key prefixes that `PoseHRNet` saved during training, trying to do `model.load_state_dict(state_dict)` crashed. 

To bypass the crash, we originally stripped the `'backbone.'` strings and used:
```python
model.load_state_dict(state_dict, strict=False)
```

**Why this broke the model completely:**
`strict=False` told PyTorch to completely ignore any weights inside the checkpoint that didn't match the new model. Because the custom `final_head` wasn't part of the official HRNet model structure, the critical, finely-trained layers responsible for actually determining exactly where those 8 points go were **completely thrown away**. The official model retained the pretrained backbone weights, but its randomly-initialized final output layer had never been trained!

Consequently, the model was outputting raw statistical noise right at the end—explaining why the predictions were awful despite perfect bounding box crops and subpixel rendering.

### 3. The Fix
To fix inference properly, all environments mapping the model must share exactly the same neural network graph logic. 

We brought your exact `PoseHRNet` class directly into `inference.ipynb`:
```python
# Exact same skeleton from train.ipynb
class PoseHRNet(nn.Module):
    def __init__(self, num_keypoints=8):
        # ...
        self.backbone = timm.create_model('hrnet_w18', ...)
        self.final_head = nn.Sequential(...)
        # ...
```

By ensuring the class perfectly matches what was trained, we can now use `strict=True` to confidently ensure *all* learned weights (both `backbone` and `final_head`) are properly activated.