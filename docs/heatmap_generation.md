### Phương pháp tạo Ground-Truth Heatmap từ Tọa độ (Heatmap Generation)

Trong bài toán nhận dạng điểm đặc trưng (Landmark Detection) sử dụng các kiến trúc mạng như HRNet hay UNet, thay vì ép mô hình hồi quy (regression) trực tiếp ra tọa độ dạng số $(x, y)$, mô hình sẽ được huấn luyện để dự đoán một tập hợp các **ảnh nhiệt (Heatmaps)**. Mỗi điểm đặc trưng (landmark) sẽ có một ảnh nhiệt riêng. 

Quy trình chuyển đổi một tọa độ landmark trên ảnh X-quang thành ảnh nhiệt Ground-Truth (được định nghĩa trong hàm `draw_gaussian`) diễn ra qua các bước sau:

#### 1. Ánh xạ tọa độ (Coordinate Scaling)
- Kích thước của Heatmap thường nhỏ hơn kích thước ảnh gốc để tối ưu hóa bộ nhớ và tính toán (Ví dụ: Ảnh gốc $512 \times 512$, Heatmap $128 \times 128$).
- Tọa độ $(x, y)$ ban đầu của điểm landmark trên ảnh gốc sẽ được nhân với tỷ lệ thu nhỏ để tìm ra tọa độ tâm $(h_x, h_y)$ tương ứng trên không gian Heatmap:
  $$ h_x = x \times \frac{W_{heatmap}}{W_{image}} $$
  $$ h_y = y \times \frac{H_{heatmap}}{H_{image}} $$

#### 2. Phủ phân phối chuẩn Gaussian 2 chiều (2D Gaussian Splatting)
Thay vì đánh dấu duy nhất một pixel tại $(h_x, h_y)$ với giá trị 1 và phần còn lại là 0 (Hard-label), chúng ta tạo ra một "đốm sáng" lan tỏa để cung cấp tín hiệu huấn luyện mượt mà hơn. 
- Tại vùng xung quanh tọa độ tâm $(h_x, h_y)$, thuật toán áp dụng hàm phân phối chuẩn 2 chiều (2D Gaussian).
- Giá trị pixel tại vị trí $(x, y)$ bất kỳ xung quanh tâm được tính bằng công thức:
  $$ G(x, y) = \exp \left( - \frac{(x - h_x)^2 + (y - h_y)^2}{2\sigma^2} \right) $$
- **Tham số $\sigma$ (Sigma):** Quy định độ "rộng" của đốm nhiệt (trong source code hiện tại đang thiết lập $\sigma = 2.5$). Tâm đốm nhiệt luôn có giá trị bằng $1$, và cường độ sẽ giảm dần về $0$ theo hình tròn khi càng xa tâm.

#### 3. Giới hạn vùng tính toán (Bounding & Truncation)
- Để tránh lãng phí tài nguyên tính toán khi tính hàm $\exp()$ cho toàn bộ ảnh, thuật toán chỉ giới hạn việc tạo ma trận Gaussian trong một hình vuông nhỏ (bounding box) có kích thước bằng bán kính $3\sigma$ xung quanh tâm landmark.
- Nếu điểm landmark nằm sát mép hoặc bị văng ra khỏi giới hạn biên của Heatmap (sau khi augmentation data), hàm sẽ cắt bỏ phần đốm nhiệt tràn ra ngoài để không gây lỗi `Out Of Bounds`.

#### Ý nghĩa của phương pháp Heatmap:
- **Tối ưu hàm Loss:** Việc tính lỗi (Loss) trên bề mặt không gian (spatial) giúp mô hình nhận được tín hiệu định hướng rõ ràng. Nếu mô hình đoán sai vài pixel, nó vẫn nhận được một phần thưởng (gradient) do nằm ở phần rìa của đốm nhiệt, từ đó dần dần được "dẫn đường" trượt về phía đỉnh tâm (peak).
- **Hạn chế Overfitting:** Giúp mạng nơ-ron học được các đặc trưng ngữ cảnh xung quanh điểm mốc thay vì phải học thuộc lòng một điểm tọa độ cứng nhắc duy nhất.
