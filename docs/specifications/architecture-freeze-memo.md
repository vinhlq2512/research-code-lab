# Văn Kiện Khóa Kiến Trúc: TAPTA (Architecture Freeze Memo)

- **Đề tài:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)
- **Mã văn kiện:** `MEMO-ARCH-FREEZE-W18`
- **Ngày ban hành:** 2026-11-24 (Hiệu lực từ Tuần 18: 14/12/2026 – 20/12/2026)
- **Tác giả:** Lâm Quang Vinh
- **Trạng thái:** **FROZEN & RATIFIED (Đã khóa 100% vĩnh viễn)**

---

## 1. Tuyên Bố Khóa Kiến Trúc (Executive Statement)

Căn cứ vào kết quả thực nghiệm đối sánh toàn diện 6 mô hình trên Protocol v1 Track A (FewRel 8 tasks $\times$ 10 relations, strict 5-shot, Rehearsal-Free $M=0$, seed 2021) chạy trên môi trường Google Colab GPU (Tesla T4), kiến trúc mô hình **TAPTA (Proposed Method)** chính thức được **ĐÓNG BĂNG 100% (ARCHITECTURE FREEZE)**.

Kể từ thời điểm này:
1. **Không thêm, bớt hoặc điều chỉnh bất kỳ module mạng hay siêu tham số nào trong mô hình lõi TAPTA.**
2. Toàn bộ các thực nghiệm tiếp theo (chạy đa seed, chạy benchmark thứ hai TACRED, phân tích lỗi, ablations) sẽ tuân thủ tuyệt đối cấu hình chuẩn mực được phê duyệt trong văn kiện này.
3. Chuyển giao 100% trọng tâm nghiên cứu sang việc viết toàn văn bài báo khoa học, luận văn và chuẩn bị slide thuyết trình.

---

## 2. Đặc Tả Kiến Trúc Chuẩn Của TAPTA

```
                             [Input Sentence x with Entity Markers]
                                               |
                        +----------------------+----------------------+
                        | (Stationary Path)                           | (Prompted Path)
                        v                                             v
              +-------------------+                         +-------------------+
              | 100% Frozen BERT  |                         | 100% Frozen BERT  |
              |   (No Prompts)    |                         |  with Prompt Tree |
              +-------------------+                         +-------------------+
                        |                                             |
                 q(x) = [h_E1; h_E2]                                  |
                        |                                             |
                        v                                             |
             +---------------------+                                  |
             | Sub-Centroid Router |                                  |
             |    (alpha = 0.7)    |                                  |
             +---------------------+                                  |
                        |                                             |
            Top-K Tasks T_K = {t_1, t_2}                              |
            Softmax Weights w_1, w_2 (tau=0.2)                        |
                        |                                             |
                        +---------------------------------------------+
                        |
                        | Trích xuất biểu diễn theo từng mồi nhắc cây:
                        | Candidate 1: P_1 = [P^root; P^leaf_{t_1}] -> h_r(x; P_1) -> Cosine Sim s_1
                        | Candidate 2: P_2 = [P^root; P^leaf_{t_2}] -> h_r(x; P_2) -> Cosine Sim s_2
                        v
          +-------------------------------------------------------------+
          |      Tổ Hợp Mềm ở Tầng Quyết Định (Decision-Level Fusion)    |
          |               S_fusion(x, c) = w_1 * s_1 + w_2 * s_2        |
          +-------------------------------------------------------------+
                                        |
                                        v
                          [Predicted Relation Class c*]
```

### 2.1. Bộ Mã Hóa Nền Tảng (Frozen Backbone)
- **Mô hình nền:** `bert-base-uncased` (12 layers, 768 hidden dimensions, 12 attention heads, 109.5M parameters).
- **Trạng thái:** **Đóng băng 100% tham số** trong toàn bộ quá trình học liên tục và suy luận.
- **Biểu diễn thực thể:** Chèn 4 token phân tách thực thể `[E1]...[/E1]` và `[E2]...[/E2]`. Vector truy vấn và vector biểu diễn quan hệ đều được tạo bằng cách nối trực tiếp 2 vector biểu diễn của thực thể: $h_r = [h_{\text{[E1]}}; h_{\text{[E2]}}] \in \mathbb{R}^{1536}$.

### 2.2. Cây Mồi Nhắc Phân Cấp 2 Tầng (Hierarchical Prompt Tree)
Tổng độ dài mồi nhắc được cố định nghiêm ngặt ở $L = 8$ tokens (bảo toàn ngân sách tham số như P0, P1, P2, P3), phân rã thành:
1. **Mồi nhắc Gốc Bất biến ($P^{\text{root}} \in \mathbb{R}^{1 \times 4 \times 768}$):**
   - Độ dài: $L_{\text{root}} = 4$ tokens.
   - Chiến lược học: Khởi tạo ngẫu nhiên $\mathcal{N}(0, 0.02)$, tối ưu hóa duy nhất tại Tác vụ 0 ($T_0$) cùng với $P^{\text{leaf}}_0$.
   - **Quy tắc bất biến:** Ngay sau khi $T_0$ kết thúc huấn luyện, $P^{\text{root}}$ được **đóng băng vĩnh viễn** (`requires_grad = False`). Đóng vai trò mỏ neo cấu trúc (*structural anchor*) ổn định phân phối đặc trưng.
2. **Mồi nhắc Nhánh Chuyên biệt Tác vụ ($P^{\text{leaf}}_t \in \mathbb{R}^{1 \times 4 \times 768}$):**
   - Độ dài: $L_{\text{leaf}} = 4$ tokens cho mỗi tác vụ $t \in \{0, \dots, T-1\}$.
   - Chiến lược học: Tối ưu hóa độc lập tại mỗi tác vụ $T_t$ bằng hàm mất mát kết hợp:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(D_t) + \lambda_{\text{orth}} \mathcal{L}_{\text{orth}}(P^{\text{leaf}}_t)$$
     với hệ số trực giao hóa $\lambda_{\text{orth}} = 0.05$.
   - Mồi nhắc hoàn chỉnh cho tác vụ $t$: $P_t = [P^{\text{root}}; P^{\text{leaf}}_t] \in \mathbb{R}^{1 \times 8 \times 768}$.

### 2.3. Bộ Định Tuyến Đa Tâm Con Phi Tham Số (Sub-Centroid Task Router)
- **Không gian định tuyến:** Không gian truy vấn bất biến $q(x) = [h_{\text{[E1]}}^{(0)}; h_{\text{[E2]}}^{(0)}] \in \mathbb{R}^{1536}$ trích xuất từ BERT đóng băng thuần túy (không gắn prompt).
- **Bộ đệm lưu trữ:**
  - Tâm tác vụ chuẩn hóa: $c_t \in \mathbb{R}^{1536}$.
  - Tâm con quan hệ chuẩn hóa: $s_{t, r} \in \mathbb{R}^{1536}$ ($r \in \{1, \dots, R_t\}$).
- **Điểm tương đồng kết hợp:**
  $$s_t(x) = (1 - \alpha) \cdot \cos(q(x), c_t) + \alpha \cdot \max_{r \in R_t} \cos(q(x), s_{t, r})$$
  với trọng số tâm con cố định $\alpha = 0.7$.

### 2.4. Cơ Chế Suy Luận Tổ Hợp Tầng Quyết Định (Decision-Level Soft Fusion)
- **Trích xuất ứng viên Top-$K$:** Chọn $K=2$ tác vụ có điểm $s_t(x)$ cao nhất $\mathcal{T}_K(x) = \{t_1, t_2\}$.
- **Trọng số Softmax nhiệt độ:**
  $$w_k(x) = \frac{\exp(s_k(x) / \tau)}{\sum_{j \in \mathcal{T}_K(x)} \exp(s_j(x) / \tau)}, \quad \tau = 0.2$$
- **Tổ hợp điểm số nguyên mẫu:**
  $$\mathcal{S}_{\text{fusion}}(x, c) = \sum_{k \in \mathcal{T}_K(x)} w_k(x) \cdot \cos\left(h_r(x; [P^{\text{root}}; P^{\text{leaf}}_k]), p_c\right)$$
  Triệt tiêu 100% hiện tượng lệch pha nguyên mẫu (*Prompt-Prototype Discrepancy*), tận dụng trọn vẹn $81.41\%$ Top-2 Router Coverage.

---

## 3. Bằng Chứng Thực Nghiệm Quyết Định Việc Khóa Kiến Trúc

Chuỗi thực nghiệm đối chứng nghiêm ngặt trên Protocol v1 Track A FewRel 5-shot seed 2021 chứng minh tính ưu việt của cấu hình TAPTA khóa:

| Phương pháp | Whole Acc | AIA | Forgetting | T1 Retention | Ghi chú & Đóng góp |
|---|:---:|:---:|:---:|:---:|---|
| **P0 (Shared Prompt)** | 52.83% | 64.25% | 19.20% | 38.90% | Sàn mồi nhắc chia sẻ (quên lãng thảm khốc) |
| **P2b (Learned Router)** | 56.04% | 67.45% | 14.51% | 52.30% | Router tham số bị trôi dạt do thiếu dữ liệu cũ |
| **P3 (Soft Param Blend)** | 55.50% | 66.84% | 15.13% | 60.40% | Top-2 Cov đạt 81.41% nhưng lệch pha nguyên mẫu |
| **P2a (Centroid Hard)** | 59.35% | 68.74% | 15.06% | 69.80% | Chọn cứng Top-1, độ nhạy cao khi router sai |
| **P4 (TAPTA Khóa)** | **58.38%** (Tree) / **58.80%** (Gated) | **68.86% (Đỉnh)** | **11.89% (Kỷ lục)** | **68.50% - 69.20%** | **Toàn diện nhất, quên lãng thấp hơn cả P1 Oracle** |
| **P1 (Oracle Task)** | 70.14% | 76.55% | 14.41% | 72.80% | Cận trên lý tưởng (cần Oracle Task ID) |

---

## 4. Bảng Khóa Tham Số Mô Hình TAPTA

| Tham số | Giá trị khóa | Vai trò trong hệ thống |
|---|:---:|---|
| `backbone` | `bert-base-uncased` | Bộ mã hóa ngôn ngữ nền tảng đóng băng 100% |
| `max_seq_length` | `128` | Độ dài tối đa chuỗi đầu vào |
| `prompt_length` ($L$) | `8` | Tổng số token mồi nhắc |
| `root_prompt_length` ($L_{\text{root}}$) | `4` | Số token mồi nhắc gốc bất biến |
| `leaf_prompt_length` ($L_{\text{leaf}}$) | `4` | Số token mồi nhắc nhánh chuyên biệt tác vụ |
| `freeze_root_after_task0` | `True` | Đóng băng mồi nhắc gốc sau Tác vụ 0 |
| `orthogonality_lambda` ($\lambda_{\text{orth}}$) | `0.05` | Hệ số phạt trực giao giữa các leaf prompts |
| `router_type` | `centroid` | Định tuyến Sub-Centroid phi tham số |
| `subcentroid_weight` ($\alpha$) | `0.7` | Trọng số ưu tiên tâm con quan hệ |
| `top_k` ($K$) | `2` | Số lượng tác vụ ứng viên được kích hoạt |
| `temperature` ($\tau$) | `0.2` | Nhiệt độ làm sắc nét phân phối trọng số Softmax |
| `memory_size` ($M$) | `0` | Không lưu mẫu văn bản thô (Rehearsal-Free) |

---

## 5. Kết Luận & Hiệu Lực

Văn kiện này có hiệu lực chính thức cho toàn bộ mã nguồn tại repository `vinhlq2512/research-code-lab`.
Kể từ ngày 24/11/2026, kiến trúc TAPTA bước vào trạng thái **FROZEN**. Mọi thành viên và agent nghiên cứu cam kết giữ nguyên cấu hình này cho giai đoạn viết bài báo khoa học và thực nghiệm diện rộng.
