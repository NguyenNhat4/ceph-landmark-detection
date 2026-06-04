# Data Processing (Cephalometric Landmark Training)

Tài liệu này mô tả chi tiết quy trình xử lý dữ liệu trong notebook huấn luyện HRNet.

## 1) Cấu trúc dữ liệu

- Dữ liệu ảnh nằm trong:
  - data/train/Cephalograms/
  - data/valid/Cephalograms/
  - data/test/Cephalograms/
- Nhãn (landmarks) nằm trong:
  - data/<split>/Annotations/Cephalometric Landmarks/<annotator>/
- File ánh xạ pixel size (mm/px):
  - data/cephalogram_machine_mappings.csv

Mỗi ảnh sẽ có một file JSON annotation cùng tên (stem). Ví dụ:
- 0001.png -> 0001.json

## 2) Đọc ảnh và annotation

Trong lớp CephLandmarkDataset:

1. Đọc ảnh bằng OpenCV:
   - cv2.imread(..., cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
   - Chuyển BGR -> RGB
2. Đọc JSON annotation:
   - Lấy danh sách landmarks: symbol, value.x, value.y
   - Lọc theo TARGET_LANDMARKS (đã loại trùng và giữ thứ tự)
3. Kiểm tra:
   - Ảnh và annotation phải tồn tại
   - Các landmark bắt buộc phải có đủ

## 3) Augmentation (chỉ train)

Nếu is_train=True, áp dụng các phép biến đổi sau:

1. Rotation + Scale:
   - Góc quay ngẫu nhiên: [-ROT_DEG, ROT_DEG]
   - Tỉ lệ scale ngẫu nhiên: 1 +- SCALE_RANGE
   - Áp dụng cùng phép biến đổi cho ảnh và tọa độ landmark
2. Intensity Jitter:
   - Tăng/giảm độ sáng và tương phản nhẹ
   - Ảnh được clip về [0, 255]

Lưu ý: các split valid/test không dùng augmentation.

## 4) Resize và chuẩn hóa tọa độ

Ảnh được resize về IMAGE_SIZE (mặc định 512x512). Khi đó:

- sx = out_w / orig_w
- sy = out_h / orig_h
- pts_resized[:, 0] *= sx
- pts_resized[:, 1] *= sy

Các hệ số sx, sy được lưu trong meta["resize_factors"] để phục vụ việc quy đổi tọa độ về ảnh gốc khi đánh giá.

## 5) Tạo heatmap target

Mỗi landmark được chuyển thành một heatmap 2D:

- Heatmap size: HEATMAP_SIZE (mặc định 128x128)
- Tỉ lệ chuyển tọa độ từ ảnh resize sang heatmap:
  - hm_x_ratio = hm_w / out_w
  - hm_y_ratio = hm_h / out_h
- Với mỗi landmark:
  - hx = x * hm_x_ratio
  - hy = y * hm_y_ratio
  - Vẽ Gaussian centered tại (hx, hy) với sigma

Nếu landmark nằm ngoài khung heatmap, target_weight cho landmark đó sẽ được set = 0.

## 6) Chuẩn hóa ảnh

Ảnh sau resize được:

1. Chuyển về [0, 1]
2. Chuẩn hóa theo mean/std ImageNet:
   - mean = [0.485, 0.456, 0.406]
   - std  = [0.229, 0.224, 0.225]
3. Đổi shape về [C, H, W]

## 7) Meta info trả ra

Dataset trả về:

- image: tensor float32, shape [3, H, W]
- target: heatmaps, shape [J, Hm, Wm]
- target_weight: shape [J, 1]
- meta:
  - image_id
  - coords_original: tọa độ landmark gốc (trước resize)
  - resize_factors: [sx, sy]
  - pixel_size_mm: mm/px từ CSV nếu có

## 8) Quy đổi heatmap -> tọa độ ảnh gốc (đánh giá)

Khi model output heatmaps:

1. Argmax để lấy (hx, hy) trên heatmap
2. Quy đổi về tọa độ ảnh resize:
   - x_img = hx * (img_w / hm_w)
   - y_img = hy * (img_h / hm_h)
3. Quy đổi về ảnh gốc:
   - x_orig = x_img / sx
   - y_orig = y_img / sy

Công thức này được dùng trong validate() để tính lỗi theo pixel hoặc mm.

## 9) Tóm tắt pipeline

1. Read image + annotation
2. (Train) Augment
3. Resize image
4. Scale landmark coords
5. Generate heatmaps + target weights
6. Normalize image
7. Return tensors + meta
