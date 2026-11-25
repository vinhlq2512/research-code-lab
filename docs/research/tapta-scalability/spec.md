# Đặc Tả Kỹ Thuật Nghiên Cứu Mở Rộng Dữ Liệu (Specification)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Tài liệu:** `docs/research/tapta-scalability/spec.md`  
**Phiên bản:** v1.0 (Frozen Architecture P4)  
**Mục tiêu:** Mở rộng nghiên cứu khả năng mở rộng dữ liệu (Data Scalability) từ Few-Shot (5-shot) đến High-Resource (Full-Data) trong điều kiện Rehearsal-Free ($M=0$) và Backbone BERT đóng băng 100%.

---

## 1. Phạm Vi Nghiên Cứu & Ranh Giới Kỹ Thuật

### 1.1. Thành phần Giữ Nguyên Tuyệt Đối (Frozen / Invariant Components)
1. **Kiến trúc Mô hình TAPTA P4:**
   - Backbone: `bert-base-uncased` (109.5M tham số), đóng băng 100% (`requires_grad = False`).
   - Cây mồi nhắc 2 tầng (*Two-Tier Hierarchical Prompt Tree*):
     - Mồi nhắc gốc: $P^{\text{root}} \in \mathbb{R}^{L_{\text{root}} \times d}$ với $L_{\text{root}} = 4, d = 768$. Huấn luyện ở Task 0 và đóng băng vĩnh viễn ở Task 1..7.
     - Mồi nhắc nhánh: $P_t^{\text{leaf}} \in \mathbb{R}^{L_{\text{leaf}} \times d}$ với $L_{\text{leaf}} = 4, d = 768$. Mỗi task huấn luyện độc lập $3,072$ tham số kèm hàm mất mát trực giao $\mathcal{L}_{\text{orth}}$.
   - Vector mồi nhắc ghép: $P_t = [P^{\text{root}} \,\|\, P_t^{\text{leaf}}] \in \mathbb{R}^{8 \times 768}$.
2. **Bộ Định Tuyến Tác Vụ Phi Tham Số (Sub-Centroid Task Router):**
   - Vector truy vấn bất biến: $q(x) = \text{Normalize}_{\ell_2}([h_{\text{[E1]}}; h_{\text{[E2]}}]) \in \mathbb{R}^{1536}$ trích xuất từ BERT đóng băng không prompt.
   - Điểm số tác vụ: $s_t(x) = (1-\alpha) q(x)^\top \mu_t + \alpha \max_{c \in R_t} q(x)^\top \nu_{t,c}$ với $\alpha = 0.7$.
3. **Cơ Chế Suy Luận & Dung Hợp Mềm Cấp Quyết Định (Decision Fusion):**
   - Không sử dụng Oracle Task ID tại inference chính thức.
   - Dung hợp mềm xác suất dự đoán Top-$K$ nhánh ($K=2, \tau=0.2$ hoặc $1.0$).
4. **Ràng Buộc Bộ Nhớ Rehearsal-Free ($M=0$):**
   - Không lưu trữ bất kỳ câu văn thô nào của các tác vụ cũ.
5. **Tập Kiểm Thử Chuẩn (Test Set):**
   - Giữ nguyên 100% tập test 8,000 mẫu của FewRel Track A ($100 \text{ mẫu/quan hệ} \times 80 \text{ quan hệ}$) để đảm bảo tính so sánh tương thích tuyệt đối với kết quả Few-Shot E012.

### 1.2. Thành Phần Mở Rộng (Scalability Extensions)
1. **Bộ nạp Dữ liệu Mở rộng (`src/data/scalable_fewrel_loader.py`):**
   - Khai thác kho dữ liệu gốc 56,000 mẫu từ `paper-reproductions/WAVE-CRE/datasets/data_with_marker.json`.
   - Hỗ trợ các mức kích thước huấn luyện cấu hình qua file yaml:
     - `train_shot = 5` (DS-05: 400 mẫu train)
     - `train_shot = 10` (DS-10: 800 mẫu train)
     - `train_shot = 50` (DS-50: 4,000 mẫu train)
     - `train_shot = 100` (DS-100: 8,000 mẫu train)
     - `train_shot = "full"` (DS-FULL: toàn bộ pool train sạch ~420 mẫu/quan hệ = 33,600 mẫu train).
2. **Thuật Toán Lấy Mẫu Lồng Nhau Xác Định (Deterministic Nested Sampling):**
   - Với mỗi relation $c$ và seed $S$, thứ tự xáo trộn danh sách mẫu huấn luyện $I_c^{(S)}$ là cố định.
   - Đảm bảo tính chất tập con nghiêm ngặt:
     $$S_5(c) \subset S_{10(c)} \subset S_{50}(c) \subset S_{100}(c) \subset S_{\text{Full}}(c)$$
3. **Triệt Tiêu Data Leakage (Zero Leakage Enforcement):**
   - Lọc bỏ toàn bộ các câu xuất hiện trong test set (dựa trên normalized token hash) trước khi tạo train pool.
   - Khẳng định $\text{Train Pool} \cap \text{Test Set} = \emptyset$.
4. **Hạ Tầng Đo Lường & Báo Cáo Tài Nguyên Mở Rộng:**
   - Ghi nhận chi tiết thời gian huấn luyện từng task, peak GPU memory, độ trễ suy luận và kích thước byte lưu trữ thực tế.
   - Thêm chế độ chạy chẩn đoán Oracle Routing (`--oracle_diagnostic`) để đo lường khoảng cách `Routing Gap`.

---

## 2. Thiết Kế Luồng Xử Lý (Pipeline Architecture)

```
[ WAVE-CRE data_with_marker.json (56,000 mẫu) ]
                       │
                       ▼
    [ LỌC BỎ TEST SET 8,000 MẪU (Zero Leakage) ]
                       │
                       ▼
        [ SẠCH TRAIN POOL (420 mẫu/quan hệ) ]
                       │
      ┌────────────────┴────────────────┐
      ▼ (Deterministic Shuffle with Seed)
 [ NESTED SAMPLING: S_5 ⊂ S_10 ⊂ S_50 ⊂ S_100 ⊂ S_Full ]
                       │
                       ▼
    [ TẬP HUẤN LUYỆN TỪNG TÁC VỤ (Tasks 1..8) ]
                       │
                       ▼
 [ HUẤN LUYỆN TAPTA P4 (Rehearsal-Free M=0, Frozen BERT) ]
   - Task 0: Học P^root (L=4) + P_0^leaf (L=4)
   - Task 1..7: P^root ĐÓNG BĂNG, chỉ học P_t^leaf (L=4) + L_orth
   - Cập nhật Relation Prototypes p_c và Router Centroids μ_t, ν_{t,c}
                       │
                       ▼
 [ ĐÁNH GIÁ TÍCH LŨY TRÊN TOÀN BỘ QUAN HỆ ĐÃ HỌC ]
   - Chế độ 1: Task-Agnostic Soft Fusion (Inference chính thức)
   - Chế độ 2: Oracle-Task Routing (Chẩn đoán Routing Gap)
                       │
                       ▼
 [ KẾT XUẤT ARTIFACTS: Metrics, Heatmaps, Performance Matrix, Manifest ]
```

---

## 3. Tiêu Chuẩn Tái Lập (Reproducibility Standard)

Mỗi experiment run phải lưu đủ bộ artifacts tại:
`runs/tapta_scalability/fewrel_{shot}shot/seed_{seed}/`
Gồm:
- `config.yaml`: Cấu hình đầy đủ.
- `dataset_manifest.json`: Fingerprint SHA-256 và danh sách ID các mẫu train/test của từng task.
- `metrics.json`: Ma trận hiệu năng $A_{t,j}$, Whole Acc, AIA, Forgetting, Router Metrics.
- `performance_matrix.csv`: Bảng ma trận $8 \times 8$.
- `resource_usage.json`: Thời gian huấn luyện, GPU type, peak memory, persistent memory (bytes).
- `summary.json`: Tóm tắt tổng kết.
