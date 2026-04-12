
1. **Setup & Data Processing**: 
   - Imports dependencies like `torch` and `timm`.
   - Creates the `CephalometricDataset` class which reads `coco_annotations.json`, crops using the bounding box, resizes the images to $256 \times 256$, and safely scales the annotations.
   - Generates $64 \times 64$ target Gaussian Heatmaps (standard 1/4 resolution for PyTorch HRNets).
   - Includes a cell to structurally verify the PyTorch data loader by drawing bounding boxes and Gaussian heatmaps on an overlay.

2. **HRNet Model wrapper using `timm`**:
   - Uses `timm.create_model('hrnet_w18', features_only=True)`. Using `features_only=True` prevents the classification pooling head from running, and exposes the intermediate network stages.
   - Extracts the first element of the high-res representation (stride 4, $64 \times 64$ size outputs).
   - Slaps a $1 \times 1$ Convolution layer on top to adjust the feature maps into exactly **8 channels** (one for each keypoint).

3. **Training Loop**:
   - Compiles a standard iteration loop using **AdamW** and **MSE Loss**.
   - I've configured it to trace and heavily penalize heatmap variances by scaling the MSE loss.

4. **Inference & Visualization**:
   - Evaluates a sample with the trained model (`model.eval()`).
   - Slices out the `argmax` from the $(64 \times 64)$ spatial channels and mathematically scales it back ($ \times 4.0$) to original image dimension $256 \times 256$.
   - Displays a 3-window overlay of Ground Truth (Red X) vs Prediction (Green Circle).

