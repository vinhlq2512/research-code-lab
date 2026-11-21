# Đặc Tả Kỹ Thuật: P2 (Task-Aware Prototype & Learned Router)

- **Mã thực nghiệm:** `P2a` (Centroid Cosine Router), `P2b` (Lightweight Learned Router)
- **Thuộc đề tài:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)
- **Giao thức chuẩn:** Protocol v1 — Track A (FewRel 8 tasks $\times$ 10 relations, strict 5-shot, Rehearsal-Free $M=0$)
- **Hạ tầng thực thi chuẩn:** Google Colab (NVIDIA GPU CUDA)
- **Ngày lập:** 2026-11-21
- **Tác giả:** Lâm Quang Vinh

---

## 1. Mục Đích & Định Vị Trong Lộ Trình TAPTA

Sau khi xác lập hai cận đối chứng ở Tuần W14 trên Protocol v1 Track A:
- **P0 (Shared Prompt — Cận dưới):** Prototype Acc = `52.83%`, TIL Acc = `82.72%`
- **P1 (Oracle-Task Prompt — Cận trên lý tưởng):** Prototype Acc = `70.14%`, TIL Acc = `82.75%`
- **Khoảng cách giá trị Task-Awareness:** $\Delta_{\text{oracle}} = +17.31\%$

Mô hình **P2** là bước chuyển tiếp cốt lõi từ cận đối chứng sang phương pháp đề xuất thực tế: **loại bỏ hoàn toàn Oracle Task ID lúc suy luận (Class-Incremental Learning chuẩn)** và thay bằng một **Bộ định tuyến tác vụ (Task Router)** tự động dự đoán $\hat{t} \in \{0, \dots, t_{\text{cur}}\}$ cho từng câu kiểm thử $x$, từ đó kích hoạt mồi nhắc $P_{\hat{t}}$ tương ứng trong `prompt_pool`.

### Độ đo đánh giá trọng tâm của P2
1. **Router Accuracy ($R_{\text{acc}}$):** Tỷ lệ mẫu kiểm thử được gán đúng Task ID thực tế $t^*$:
   $$R_{\text{acc}} = \frac{1}{|D_{\text{test}}|} \sum_{(x, y) \in D_{\text{test}}} \mathbb{I}\left(\hat{t}(x) = t^*(y)\right)$$
2. **Whole Prototype Accuracy ($A_T^{\text{proto}}$):** Độ chính xác phân loại quan hệ trên toàn bộ tập nhãn đã học $\widetilde{R}_T$ mà không dùng Task ID.
3. **Routed-Gated Prototype Accuracy ($A_T^{\text{gated}}$):** Độ chính xác phân loại khi kết hợp định tuyến tác vụ $\hat{t}$ và khoanh vùng không gian nguyên mẫu thuộc tác vụ được chọn $R_{\hat{t}}$.
4. **Tỷ lệ thu hẹp khoảng cách trần - sàn (Oracle Gap Closed):**
   $$\text{Gap Closed (\%)} = \frac{A_T(P2) - A_T(P0)}{A_T(P1) - A_T(P0)} \times 100\%$$
5. **Ma trận nhầm lẫn định tuyến (Task Confusion Matrix $C \in \mathbb{R}^{8 \times 8}$):** $C_{i, k}$ ghi nhận tỷ lệ mẫu của tác vụ thực $i$ bị định tuyến sang tác vụ $k$.

---

## 2. Trích Xuất Biểu Diễn Truy Vấn Bất Biến (Zero-Drift Query Representation)

Để định tuyến một câu $x$ trước khi biết cần ghép mồi nhắc $P_t$ nào, P2 khai thác đặc tính **đóng băng 100% (Frozen Backbone)** của encoder BERT $\mathcal{E}_{\text{frozen}}$:

1. Đưa câu đã gắn 4 thẻ thực thể $x_{\text{marked}}$ đi qua $\mathcal{E}_{\text{frozen}}$ **không kèm soft prompt**:
   $$H^{(0)} = \mathcal{E}_{\text{frozen}}(x_{\text{marked}}) \in \mathbb{R}^{N \times d} \quad (d = 768)$$
2. Trích xuất vector trạng thái ẩn tại vị trí hai thẻ mở thực thể `[E1]` và `[E2]`:
   $$q_{e1} = H^{(0)}[\text{pos}(\text{[E1]})], \quad q_{e2} = H^{(0)}[\text{pos}(\text{[E2]})]$$
3. Nối và chuẩn hóa $\ell_2$ để thu được **vector truy vấn định tuyến (Query Vector)**:
   $$q(x) = \text{Normalize}_{\ell_2}\left([q_{e1}; q_{e2}]\right) \in \mathbb{R}^{2d} \quad (2d = 1536)$$

> **Tính chất bất biến (Zero Representation Drift):** Vì $\mathcal{E}_{\text{frozen}}$ không bao giờ cập nhật trọng số qua các tác vụ $T_1 \dots T_8$, phân phối của $q(x)$ hoàn toàn cố định theo thời gian. Mọi thống kê hoặc nguyên mẫu tính trên không gian $q(x)$ ở tác vụ cũ không bao giờ bị trôi dạt.

---

## 3. Hai Cơ Chế Định Tuyến Đối Chứng (P2a vs P2b)

### 3.1. P2a: Centroid Cosine Router (Training-Free / Non-Parametric)
Ở giai đoạn huấn luyện tác vụ $t$, từ tập mẫu 5-shot $D_t = \bigcup_{c \in R_t} D_{t,c}$:
- Tính **Tâm tác vụ tổng quát (Task Centroid)**:
  $$\mu_t = \text{Normalize}_{\ell_2}\left(\frac{1}{|D_t|} \sum_{x \in D_t} q(x)\right) \in \mathbb{R}^{2d}$$
- Tính **Tâm con theo từng quan hệ (Relation Sub-Centroids)** cho mỗi $c \in R_t$:
  $$\nu_{t,c} = \text{Normalize}_{\ell_2}\left(\frac{1}{|D_{t,c}|} \sum_{x \in D_{t,c}} q(x)\right) \in \mathbb{R}^{2d}$$

Vì một tác vụ chứa 10 quan hệ đa dạng ngữ nghĩa, điểm tương đồng của câu $x$ với tác vụ $t$ được xác định bằng hợp nhất tương đồng tâm tác vụ và tâm quan hệ gần nhất:
$$s_t^{\text{P2a}}(x) = (1 - \alpha) \cdot \left(q(x)^\top \mu_t\right) + \alpha \cdot \max_{c \in R_t} \left(q(x)^\top \nu_{t,c}\right)$$
(trong đó $\alpha \in [0, 1]$, mặc định $\alpha = 0.7$ ưu tiên cấu trúc đa tâm của các quan hệ trong tác vụ; khi $\alpha = 0.0$ trở về thuần túy đơn tâm $\mu_t$).

Tác vụ được chọn lúc suy luận:
$$\hat{t} = \arg\max_{t \in \{0, \dots, t_{\text{cur}}\}} s_t^{\text{P2a}}(x)$$

### 3.2. P2b: Lightweight Learned Router (MLP Cosine Router với Thống kê Đặc trưng Đóng băng)
Để học biên phân tách phi tuyến giữa các tác vụ mà vẫn tuân thủ nghiêm ngặt ràng buộc **Rehearsal-Free ($M=0$, không lưu bất kỳ câu văn thô nào)**:
1. **Kiến trúc mạng định tuyến $R_\phi$:**
   $$z_{\text{route}}(x) = \text{LN}\left(\text{GELU}(q(x) W_1 + b_1)\right) \in \mathbb{R}^{d_r} \quad (d_r = 512)$$
   $$s_t^{\text{P2b}}(x) = \tau \cdot \cos\left(z_{\text{route}}(x), k_t\right)$$
   trong đó $k_t \in \mathbb{R}^{d_r}$ là vector khóa học được (learnable task key) của tác vụ $t$ và $\tau = 16.0$ là hệ số nhiệt độ (inverse temperature).
2. **Huấn luyện chống quên trong không gian đặc trưng đóng băng ($M=0$):**
   - Khi kết thúc mỗi tác vụ $t$, lưu lại cặp vector thống kê gọn nhẹ $(\nu_{t,c}, \sigma_{t,c}^2) \in \mathbb{R}^{2d}$ của các quan hệ trong không gian bất biến $q(x)$ (tuyệt đối không lưu văn bản gốc).
   - Khi học tác vụ mới, sinh các mẫu đặc trưng giả lập $\tilde{q} \sim \mathcal{N}(\nu_{j,c}, \sigma_{j,c}^2 I)$ cho các tác vụ cũ $j < t$ và tối ưu hàm mất mát Cross-Entropy trên tập tác vụ đã thấy $\{0, \dots, t\}$:
     $$\mathcal{L}_{\text{router}} = -\log \frac{\exp(s_{t^*}^{\text{P2b}}(q))}{\sum_{j=0}^{t} \exp(s_j^{\text{P2b}}(q))}$$

---

## 4. Luồng Suy Luận Tổng Thể (Two-Stage Prompted Inference)

Với mỗi lô mẫu kiểm thử $B = \{x_1, \dots, x_m\}$ tại giai đoạn $t_{\text{cur}}$ (không có Task ID):
1. **Bước 1 (Task Routing):** Trích xuất $q(x_i)$ qua $\mathcal{E}_{\text{frozen}}$ và dự đoán $\hat{t}_i \in \{0, \dots, t_{\text{cur}}\}$ bằng bộ định tuyến (P2a hoặc P2b).
2. **Bước 2 (Task-Prompted Feature Extraction):** Ghép mồi nhắc $P_{\hat{t}_i} \in \mathbb{R}^{L \times d}$ tương ứng với từng mẫu $x_i$:
   $$E_{\text{prompted}}(x_i) = [P_{\hat{t}_i}; E_{\text{tokens}}(x_i)]$$
   Đưa qua $\mathcal{E}_{\text{frozen}}$ để trích xuất biểu diễn quan hệ đã thích ứng mồi nhắc $h_r(x_i; P_{\hat{t}_i}) \in \mathbb{R}^{2d}$.
3. **Bước 3 (Prototype Relation Classification):**
   - **Chế độ CIL Toàn cục (`prototype_accuracy`):** So khớp cosine giữa $h_r(x_i; P_{\hat{t}_i})$ với toàn bộ nguyên mẫu đã học $\widetilde{R}_{t_{\text{cur}}}$.
   - **Chế độ CIL Khoanh vùng theo Router (`routed_prototype_accuracy`):** So khớp cosine giữa $h_r(x_i; P_{\hat{t}_i})$ với các nguyên mẫu thuộc tác vụ được chọn $R_{\hat{t}_i}$.

---

## 5. Tiêu Chí Kiểm Chứng (Acceptance Criteria)

- Mã nguồn chạy hoàn chỉnh trên Google Colab GPU (CUDA) qua script `experiments/run_p2.py` và notebook `notebooks/p2_router_colab_seed2021.ipynb`.
- Xuất đầy đủ `metrics.json`, `router_confusion_matrix.json`, `prototype_performance_matrix.csv` và `routed_prototype_performance_matrix.csv`.
- **Mục tiêu định lượng:** Whole Accuracy của P2 đạt $\ge 60.0\%$ (vượt mốc sàn 52.83% của P0 và đóng tối thiểu 40% khoảng cách tới mốc trần 70.14% của P1).
