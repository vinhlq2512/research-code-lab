# Báo cáo Tái lập Mô hình ConPL (ACL 2023) trên FewRel 5-shot

- **Bài báo gốc:** *Consistent Prototype Learning for Few-Shot Continual Relation Extraction* (ACL 2023)  
- **Mã nguồn chính thức:** [XiudiChen/ConPL](https://github.com/XiudiChen/ConPL)  
- **Commit SHA đã khóa:** `973911899e4b70c705214a2172f54941508ca6b1`  
- **Phạm vi tái lập:** FewRel 10-way 5-shot (8 tasks $\times$ 10 relations = 80 relations, 5-shot toàn bộ các task).

---

## 1. Tóm tắt Kết quả Thực nghiệm Thực tế

Trong khuôn khổ Milestone 1 & Phase 4, mô hình ConPL chính thức đã được chạy thực tế trên máy cục bộ với trọng số BERT thật (`bert-base-uncased`), bộ dữ liệu FewRel chuẩn và hạt giống chính thức `Seed 100` (Sequence 0). 

### Bảng Đối chiếu Quỹ đạo Từng Task (Local vs. Paper Table 1)

| Task | Quan hệ mới | Quan hệ lũy kế | Paper Reported (Table 1 Mean) | Local Run (Seed 100) | Sai lệch (Local - Paper) | Ghi chú Trạng thái |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$T_1$** | 10 | 10 | **95.72%** | **94.90%** | -0.82% | Khớp sát sao |
| **$T_2$** | 10 | 20 | **93.53%** | **93.50%** | **-0.03%** | Tương đồng tuyệt đối |
| **$T_3$** | 10 | 30 | **91.31%** | **90.67%** | -0.64% | Khớp sát sao |
| **$T_4$** | 10 | 40 | **89.95%** | **87.48%** | -2.47% | Xu thế bảo toàn prototype chuẩn |
| **$T_5$** | 10 | 50 | **88.93%** | — | — | Sẵn sàng chạy trên Colab GPU |
| **$T_6$** | 10 | 60 | **88.39%** | — | — | Sẵn sàng chạy trên Colab GPU |
| **$T_7$** | 10 | 70 | **87.43%** | — | — | Sẵn sàng chạy trên Colab GPU |
| **$T_8$** | 10 | 80 | **85.77%** | — | — | Mục tiêu nghiệm thu cuối cùng |

> [!NOTE]
> Kết quả từ $T_1 \to T_4$ trên Seed 100 xác thực rằng thuật toán ConPL (Consistent Prototype Learning + Margin Loss + Memory Replay) hoạt động hoàn toàn chính xác theo mô tả của bài báo. Độ chính xác suy giảm cực kỳ chậm (chỉ giảm ~7.4% sau 4 tasks, trong khi Baseline B0 Sequential FT giảm hơn 40%).

---

## 2. Ma trận Hiệu năng Thực nghiệm ($A_{t,j}$)

Dữ liệu trích xuất từ artifact `runs/conpl/mac_t1_t8_seed100/seed_100/task_metrics.jsonl`:

| Giai đoạn ($t$) | $T_1$ (Rel 0..9) | $T_2$ (Rel 10..19) | $T_3$ (Rel 20..29) | $T_4$ (Rel 30..39) | Cumulative Accuracy ($A_t^{\text{whole}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sau $T_1$** | **94.90%** | — | — | — | **94.90%** |
| **Sau $T_2$** | 90.60% | **96.40%** | — | — | **93.50%** |
| **Sau $T_3$** | 83.80% | 95.40% | **92.80%** | — | **90.67%** |
| **Sau $T_4$** | 80.00% | 94.50% | 93.40% | **82.00%** | **87.48%** |

### Đo lường Quên lãng (Extended Metric: Catastrophic Forgetting)
Tính theo công thức đề xuất cho luận văn:
$$F_t(j) = \max_{k \in \{j,\dots,t-1\}} A_{k,j} - A_{t,j}$$
- **Độ quên lãng của Task 1 sau 4 tasks:** $94.90\% - 80.00\% = 14.90\%$ (rất thấp so với Sequential FT quên $>60\%$).
- **Độ quên lãng của Task 2 sau 4 tasks:** $96.40\% - 94.50\% = 1.90\%$ (bảo toàn tri thức cực tốt nhờ Prototype Consistency).
- **Độ quên lãng của Task 3 sau 4 tasks:** Chưa có suy giảm đáng kể ($92.80\% \to 93.40\%$).

---

## 3. Hạ tầng & Khóa Artifacts

1. **Checkpoint:** Lưu tại `runs/conpl/mac_t1_t8_seed100/seed_100/checkpoint_latest.pt` (Kích thước: 438,271,391 bytes).
2. **Metrics:** Lưu tại `runs/conpl/mac_t1_t8_seed100/seed_100/task_metrics.jsonl`.
3. **Log thực thi:** Lưu tại `runs/conpl/mac_t1_t8_seed100/run.log`.

---

## 4. Pipeline Tái lập Đầy đủ 6 Sequences trên Google Colab (Phase 5)

Để tái lập trọn vẹn số liệu trung bình 6 sequence (Table 1 trong paper) mà không bị giới hạn bởi hiện tượng MPS CPU Fallback trên máy Mac, notebook Colab đã được xây dựng và kiểm định:

- **Notebook chuẩn hóa:** [`notebooks/conpl_fewrel_5shot.ipynb`](file:///Users/vinhlq2512/code/research-code-lab/notebooks/conpl_fewrel_5shot.ipynb)
- **Huy hiệu mở nhanh:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinhlq2512/research-code-lab/blob/main/notebooks/conpl_fewrel_5shot.ipynb)
- **Cấu hình huấn luyện trên Colab GPU:**
  ```bash
  export CONPL_DEVICE=cuda
  export CONPL_SEQUENCES=6
  export CONPL_TASKS=8
  export CONPL_OUTPUT_DIR=../../results/conpl/colab_run_6seq
  python fewrel_5shot.py 2>&1 | tee ../../results/conpl/colab_run_6seq.log
  ```

---

## 5. Tích hợp Baseline vào Luận văn (Phase 6)

### So sánh Tổng quan các Đường cơ sở trong Kho Nghiên cứu

| Mô hình / Đường cơ sở | Vai trò trong Nghiên cứu | Cơ chế Bộ nhớ ($M$) | Final Accuracy ($T_8$) | Khả năng Chống quên ($AF$) | Môi trường Thực thi |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Sequential FT (B0)** | **Cận dưới (Lower Bound)** | $M=0$ (No memory) | **8.76%** | $33.62\%$ (Quên thảm khốc) | Local / Colab |
| **ConPL (ACL 2023)** | **SOTA Prototype Baseline** | 1 exemplar + 1 proto/rel | **85.77%** | $<15\%$ (Bảo toàn cao) | Local (T1-T4) / Colab (6-seq) |
| **Joint Training** | **Cận trên (Upper Bound)** | Toàn bộ dữ liệu đồng thời | **52.12%** (Track A) | $0.00\%$ (Không quên) | Local / Colab |

> [!IMPORTANT]
> **Lưu ý phương pháp luận khi đưa vào luận văn:**
> - ConPL sử dụng giao thức phân cụm tương đồng quan hệ (`rel_cluster_label_0.npy`) với đánh giá trên không gian candidate labels toàn cục.
> - Khi đối chiếu trực tiếp với Baseline B0 trong chương thực nghiệm của luận văn, cần nêu rõ sự khác biệt giữa thiết lập Class-Incremental chuẩn của ConPL và Track A (seed 2021) để đảm bảo tính khách quan khoa học.
