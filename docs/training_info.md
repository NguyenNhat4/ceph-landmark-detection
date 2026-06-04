### 4.1.1. Môi trường phần cứng và phần mềm
- **Cấu hình máy tính:**
  - **GPU:** NVIDIA GeForce RTX 4050 (6GB VRAM - dựa trên thông tin nvidia-smi)
  - **RAM:** (Chưa có thông tin chi tiết, có thể bổ sung sau)
- **Thư viện sử dụng:** 
  - **Deep Learning Framework:** PyTorch
  - **Các thư viện xử lý khác:** OpenCV (cv2), NumPy, Pandas, tqdm

### 4.1.2. Siêu tham số huấn luyện (Hyperparameters)
- **Hàm Loss sử dụng:** MSE Loss (Mean Squared Error) với trọng số (Weighted MSE)
- **Thuật toán tối ưu:** AdamW
- **Learning rate ban đầu (Base LR):** 2e-4 (0.0002)
- **Kích thước batch size:** 4
- **Số Epochs:** 20
- **Các siêu tham số khác (nếu cần):**
  - **Weight Decay:** 1e-4
  - **Learning Rate Scheduler:** CosineAnnealingLR
  - **Gradient Clip Norm:** 1.0
