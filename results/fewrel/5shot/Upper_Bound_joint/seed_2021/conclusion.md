# Upper Bound: Joint Training (Multitask) Empirical Report

## 1. Executive Summary
- **Benchmark:** FewRel Track A (80 relations, 5-shot, seed 2021)
- **Method:** Joint Training (Multitask Upper Bound, all 80 classes unmasked)
- **Best Validation Epoch:** Epoch 15/15 (Val Accuracy: 53.25%)
- **Overall Test Accuracy (80-way):** **52.12%**
- **Overall Test Macro-F1:** **48.62%**

## 2. Head-to-Head Comparison: Upper Bound vs Lower Bound (B0)

| Metric | Upper Bound (Joint) | Lower Bound (B0 Sequential FT) | Gap (Upper - Lower) |
| :--- | :---: | :---: | :---: |
| **Final Accuracy (AA)** | **52.12%** | 8.76% | **+43.35%** |
| **Final Macro-F1** | **48.62%** | 9.69% | **+38.93%** |
| **Catastrophic Forgetting (AF)** | **0.00%** (N/A) | 33.62% | — |
| **Average Incremental Acc (AIA)** | **52.12%** | 18.98% | **+33.13%** |

> [!NOTE]
> Joint Training đạt **52.12%** trên không gian 80 lớp, thiết lập trần hiệu năng (Upper Bound) vững chắc.
> Khoảng cách giữa Upper Bound và Lower Bound B0 là **43.35%**, đây chính là không gian tối ưu hóa (Optimization Room) cho các phương pháp Continual Learning như ConPL, CPL và TAPTA.

## 3. Per-Task Breakdown on Test Set (1,400 samples / task)

| Task ID | Task Name | Relations | Sample Count | Test Accuracy | Test Macro-F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | T1 | 10 | 6900 | 51.41% | 59.21% |
| 1 | T2 | 10 | 6900 | 51.12% | 62.36% |
| 2 | T3 | 10 | 6900 | 60.41% | 67.25% |
| 3 | T4 | 10 | 6900 | 51.54% | 62.10% |
| 4 | T5 | 10 | 6900 | 58.90% | 68.76% |
| 5 | T6 | 10 | 6900 | 63.49% | 70.91% |
| 6 | T7 | 10 | 6900 | 38.55% | 48.10% |
| 7 | T8 | 10 | 6900 | 41.25% | 47.32% |

## 4. Scientific Conclusion
1. **Phân loại 80 lớp Few-Shot (5-shot):** Dù chỉ có 5 mẫu huấn luyện cho mỗi quan hệ, việc huấn luyện đa nhiệm đồng thời cho phép biểu diễn thực thể của BERT học không gian đặc trưng toàn cục mà không bị trôi trọng số.
2. **Thước đo chuẩn mực (Benchmark Ceiling):** Mọi mô hình Continual Learning trong các nghiên cứu tiếp theo sẽ được đánh giá dựa trên mức độ tiếp cận Cận trên này so với Cận dưới B0.
