# Đặc Tả Giao Thức Chuẩn Thực Nghiệm — Protocol v1 (Locked)

- **Ngày khóa (Freeze Date):** 2026-11-14
- **Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPA-CRE / TAPTA)
- **Áp dụng cho:** Toàn bộ baselines (B0, B1, B2, B3, B4, B5) và proposed methods (P0, P1, P2, P3, P4).

---

## 1. Nguyên Tắc Cốt Lõi (Core Principles)

1. **Class-Incremental Learning:**
   - Không gian quan hệ giữa các task là rời nhau ($R_i \cap R_j = \emptyset, \forall i \neq j$).
   - Sau task $t$, mô hình phải dự đoán trên toàn bộ không gian quan hệ đã học từ trước đến nay: $\bigcup_{i=1}^t R_i$.
2. **Không có Task ID lúc Inference (No Task Oracle at Test Time):**
   - Đánh giá chính thức tuyệt đối **không** được cấp task ID tại thời điểm inference.
   - Thiết lập có Task ID (Oracle Task ID) chỉ dùng làm mốc so sánh trần (Upper bound / Ablation P1).
3. **Mô hình Backbone:**
   - `bert-base-uncased`, độ dài chuỗi tối đa `max_seq_length = 256` (hoặc 128 tùy ablation được ghi rõ), kèm theo 4 entity markers đặc biệt: `[E11]`, `[E12]`, `[E21]`, `[E22]`.
   - Mỗi baseline giữ nguyên chế độ freeze/fine-tune encoder theo đúng kiến trúc gốc, nhưng phải báo cáo số lượng `trainable_parameters`.

---

## 2. Phân Phối Dữ Liệu & Benchmark (Datasets & Splits)

Dự án khóa hai tập dữ liệu chuẩn dựa trên phân tập ConPL / CFRL:

### 2.1. FewRel (Track A — Strict Few-Shot Core)
- **Tổng số quan hệ:** 80 relations.
- **Cấu hình task:** 8 tasks liên tục, mỗi task đúng 10 relations ($8 \times 10$).
- **Số mẫu huấn luyện:** Đúng **5-shot** cho mỗi quan hệ (50 mẫu train / task). Cả 8 task đều bình đẳng về số lượng mẫu (không thiên vị task 1 data-rich).
- **Phân cụm & Task Order:** Khóa theo file cluster phân cấp ngữ nghĩa của ConPL (`CFRLdata_10_100_10_5/rel_cluster_label_0.npy`).
- **Tập kiểm thử (Test set):** 100 mẫu / relation $\times$ 10 relations = 1,000 mẫu kiểm thử / task. Tổng tích lũy sau Task 8 là 8,000 mẫu.

### 2.2. TACRED (Track A — Realistic Imbalanced)
- **Tổng số quan hệ:** 40 relations (đã loại bỏ quan hệ phủ định `no_relation`).
- **Cấu hình task:** 8 tasks liên tục, mỗi task đúng 5 relations ($8 \times 5$).
- **Số mẫu huấn luyện:** Chuẩn 5-shot cho mỗi quan hệ (25 mẫu train / task).
- **Nguồn dữ liệu:** Khóa theo bản phân tách tiền xử lý của ConPL/CFRL (`data/tacred/CFRLdata_10_100_10_5/`), ghi rõ nguồn gốc provenance.

---

## 3. Ngân Sách Bộ Nhớ (Memory Budgets)

Mọi thí nghiệm phải báo cáo ở 1 trong 2 setting hoặc cả hai:
1. **Setting $M=1$ (Main Core):** Lưu đúng 1 exemplar raw sample và 1 prototype biểu diễn cho mỗi quan hệ đã học (giống ConPL).
2. **Setting $M=0$ (Rehearsal-Free):** Hoàn toàn không lưu raw text mẫu cũ. Cho phép lưu prototype thống kê, Gaussian distribution hoặc prompt parameters, nhưng phải ghi rõ kích thước byte trong cột `Stored memory`.

---

## 4. Hệ Thống Chỉ Số Đánh Giá (Metrics)

Evaluator bắt buộc phải tính và lưu trữ ma trận hiệu năng $A \in \mathbb{R}^{T \times T}$ ($A_{t,j}$ là độ chính xác trên tập kiểm thử của task $j$ sau khi học xong task $t$):

$$A = \begin{bmatrix}
A_{1,1} & & & \\
A_{2,1} & A_{2,2} & & \\
\vdots & \vdots & \ddots & \\
A_{T,1} & A_{T,2} & \cdots & A_{T,T}
\end{bmatrix}$$

Từ ma trận $A$, tính toán các chỉ số thống nhất:
- **Final Average Accuracy ($A_T$ / Accuracy Whole):** Độ chính xác tích lũy có trọng số trên toàn bộ các relation đã học sau task cuối cùng.
- **Average Incremental Accuracy (AIA):** $\frac{1}{T} \sum_{t=1}^T \text{Acc}_t$.
- **Average Forgetting ($F_T$):** $\frac{1}{T-1} \sum_{j=1}^{T-1} \left( \max_{t \in \{j, \dots, T-1\}} A_{t,j} - A_{T,j} \right)$.
- **Backward Transfer (BWT):** $\frac{1}{T-1} \sum_{j=1}^{T-1} (A_{T,j} - A_{j,j})$.
- **Task ID Accuracy:** Tỷ lệ dự đoán chính xác task / prompt pool (đối với các mô hình router B5, P2–P4).
- **Oracle vs. Predicted Routing Gap:** Khoảng cách hiệu năng giữa khi biết task ID trước và khi phải tự predict.

---

## 5. Quy Chuẩn Lưu Trữ Artifact & Tái Lập (Reproducibility Standards)

Mỗi lần chạy thí nghiệm trong `research-code-lab` bắt buộc tạo thư mục artifact:
`runs/<model_id>/<dataset_and_setting>/seed_<seed>/`

Bao gồm tối thiểu:
1. `config.yaml`: Toàn bộ hyper-parameters và đường dẫn dữ liệu.
2. `provenance.json`: Commit hash của `research-code-lab`, commit upstream của baseline, git status diff, ngày giờ chạy.
3. `env_info.txt`: Đầu ra của `pip freeze`, phiên bản PyTorch, CUDA version, GPU model (`nvidia-smi`).
4. `run.log`: Toàn bộ log stdout và stderr trong quá trình huấn luyện và đánh giá.
5. `metrics.json`: Chứa toàn bộ ma trận $A_{t,j}$, các metric phái sinh, test sample sizes theo từng task.
6. **Checkpoints:** File `.pt` / `.ckpt` dung lượng lớn được lưu trữ trên Cloud Storage / Google Drive cá nhân, trong repo chỉ lưu đường dẫn và mã băm SHA-256.
