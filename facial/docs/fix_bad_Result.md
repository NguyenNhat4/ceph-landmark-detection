Ran terminal command:  cat << 'EOF' > test_hrnet.py
import torch
import timm
model = timm.create_model('hrnet_w18', pretrained=False)
for name, child in model.named_children():
    print(name)
EOF
python test_hrnet.py

Ran terminal command:  python test_hrnet.py
Completed with input:  python test_hrnet.py

Ran terminal command:  source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet.py

Ran terminal command:  cat << 'EOF' > test_hrnet_fwd.py
import torch
import timm
import traceback

model = timm.create_model('hrnet_w18', pretrained=False)
try:
    dummy_input = torch.randn(1, 3, 512, 512)
    x = model.conv1(dummy_input)
    print("Calling model.stages(x)...")
    out = model.stages(x)
except Exception as e:
    traceback.print_exc()
EOF
source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet_fwd.py

Ran terminal command:  source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet_fwd.py

Ran terminal command:  source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet_fwd.py 2>&1 | head -n 20

Ran terminal command:  source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet_fwd.py 2>&1 | tail -n 20

Ran terminal command:  cat /home/nhatnm/.config/Code/User/workspaceStorage/9a3578be79730b9e15de86b834054282/GitHub.copilot-chat/chat-session-resources/ce9e7772-b024-4b8a-b1a0-842adb44ad1f/call_MHx4elNMbk5URlBBZ2ZMcW5rYTg__vscode-1774187992111/content.txt | tail -n 30

Ran terminal command:  cat << 'EOF' > test_hrnet2.py
import torch
import timm

model = timm.create_model('hrnet_w18', pretrained=False)
dummy_input = torch.randn(1, 3, 512, 512)
x = model.conv1(dummy_input)
x = model.bn1(x)
x = model.act1(x)
x = model.conv2(x)
x = model.bn2(x)
x = model.act2(x)

# THIS IS THE BROKEN PART!
out = model.stages(x)
print(type(out))
EOF
source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet2.py

Ran terminal command:  source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet2.py 2> error.log && echo "SUCCESS" || echo "FAILED" ; cat error.log

Ran terminal command:  cat << 'EOF' > test_hrnet3.py
import torch
import timm
model = timm.create_model('hrnet_w18', pretrained=False)
dummy_input = torch.randn(1, 3, 512, 512)
x = model.conv1(dummy_input)
x = model.bn1(x)
x = model.act1(x)
x = model.conv2(x)
x = model.bn2(x)
x = model.act2(x)
out = model.stages(x)
for i, o in enumerate(out):
    print(f"out[{i}] shape: {o.shape}")
EOF
source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet3.py

Ran terminal command:  source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_hrnet3.py

Ran terminal command:  jq '.annotations[0]' /home/nhatnm/code/myprojects/ceph-landmark-detection/facial/coco_annotations.json

Ran terminal command:  cat << 'EOF' > test_loss.py
import torch
import torch.nn as nn
import numpy as np

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0.0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            weight = target_weight[:, idx]
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt).mean(dim=1) * weight.squeeze(-1)
        return loss.mean()

crit = WeightedMSELoss()
# simulate a target with 1.0 peak
gt = torch.zeros(1, 8, 128, 128)
gt[0, 0, 64, 64] = 1.0

# pred is all zeros initially
pred = torch.zeros(1, 8, 128, 128)
w = torch.ones(1, 8, 1)

print("Loss:", crit(pred, gt, w).item() * 1000)

EOF
source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_loss.py

Ran terminal command:  source /home/nhatnm/code/myprojects/ceph-landmark-detection/venv/bin/activate && python test_loss.py

Edit [](file:///home/nhatnm/code/myprojects/ceph-landmark-detection/facial/train.ipynb)

Edit [](file:///home/nhatnm/code/myprojects/ceph-landmark-detection/facial/train.ipynb)

There are **two major structural flaws** making your HRNet training fail or produce completely hallucinated/bad results. I have updated the notebook (`train.ipynb`) to fix them. Here is the breakdown of what was going wrong:

### 1. The Bounding Box Mismatch (The biggest problem)
In your `CephalometricDataset`, your `ann['bbox']` bounded only the keypoints (the facial profile surface: lips, nose, chin). This meant your training images were taking a tight ~300x300 pixel crop of just the facial profile and scaling it up to 512x512. The network basically learned to process extremely zoomed-in pictures of skin/bones.

However, during **inference** (Cell 7), because you lacked a bounding box, your code assumed the bounding box was the *entire image*:
```python
orig_h, orig_w = image_rgb.shape[:2]
center_w = orig_w / 2.0  # Centers on the whole cranium
# ...
scale = max(orig_w, orig_h) / 200.0 * 1.25 # Scale was almost 3x larger!
```
When HRNet was given this zoomed-out skull in inference, it had absolutely no idea what to do because it had never seen a zoomed-out global image during training.

* **The Fix I applied:** I updated Cell 3 to train on the bounding box of the **entire image** instead of `ann['bbox']`. Because medical Cephalograms are highly standardized, training the network on the whole global image context ensures that your training domain exactly matches your inference domain. 

### 2. Loss Scaling & Weight Decay Collapse 
In your `WeightedMSELoss`, calculating the error over spatial dimensions used `.mean(dim=1)`. Because a landmark heatmap is 99% zeros and just a tiny Gaussian peak, dividing the raw error by $128 \times 128$ ($16384$) resulted in gradients that were microscopically small (approx. `0.00003`).

You combined this tiny loss with **`AdamW`** and a high learning rate of `1e-3`. `AdamW` isolates weight decay directly onto the weights (default `decay=0.01`). Because the gradient signal from your loss was crushed by the spatial mean, the `0.01` weight decay massively overpowered the data gradient signal, crushing your model's weights to zero and preventing it from learning. 

* **The Fix I applied:** I updated Cell 5 to use `.sum(dim=1)` instead of `.mean(dim=1)` inside the loss function, removed the brittle `* 1000` multiplier, changed the optimizer to standard `Adam` (no isolated weight decay collapse), and reduced the Learning Rate from `1e-3` to `1e-4` (a much better LR for fine-tuning pre-trained ImageNet `timm` models).

If you re-run the notebook top-to-bottom with these adjustments, it should converge heavily and properly align with your unseen test inference to training!

Made changes.