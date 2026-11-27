# M7 Implementation Plan — DynaTree-CRE Diagnosis & Architecture Refinement

**Nguyên tắc:** Thực hiện tuần tự từng milestone; không gộp nhiều thay đổi cùng lúc; mọi thí nghiệm dữ liệu thật chạy **100% trên Google Colab GPU (`device: cuda`)**; không sửa/xóa `DT-C0..DT-C4` và artifacts cũ.

---

## 1. Lộ trình Milestones & Definition of Done (DoD)

| Milestone | Mục tiêu | Loại thí nghiệm | Definition of Done |
|---|---|---|---|
| **M7.0** | Gate 5 Re-Audit & Fix `DT-C0` TACRED | Audit + Colab `DT-C0` TACRED (`raw` & `centered`) | Có `M7_GATE5_REAUDIT.md`, sửa config/notebook TACRED `DT-C0`, xác minh resume tensor/prediction equality trên `.pt`. |
| **M7.1** | Routing Error Decomposition | Checkpoint-based diagnostic (Colab `.pt`) | Có `ROUTING_ERROR_ANALYSIS.md` đầy đủ 2×2 breakdown, `Recall@1/3/5/10`, per-cluster & calibration. |
| **M7.2** | Dual-Space Routing (`Semantic=t1_centered`, `Retrieval=raw_cosine`) | Checkpoint diagnostic trước; retrain nếu đổi policy | So sánh 2×2 (`Raw/Centered Tree × Raw/Centered Routing`) với threshold có đối chứng. |
| **M7.3** | Cluster-Size-Aware & Hybrid Routing | Diagnostic + Retrain Variants A/B/C | Pass 8 unit/invariant tests (`gamma=0` = flat, `eta=0` = `DT-C4`, synthetic equal-score); có bằng chứng thực nghiệm trên 4 phân phối. |
| **M7.4** | Candidate-Aware Prompt Reranking (`DT-C2.1`, `DT-C4.4`) | Retrain + Eval $K \in \{1,3,5,10\}$ | Chạy end-to-end không leakage/replay; xác minh `Final Acc = Recall@K × Cond Rerank Acc`; báo cáo trade-off accuracy/latency. |
| **M7.5** | Improved Retrieval Space (nếu M7.2–M7.4 chưa đủ vượt `DT-C0 Raw`) | Retrain tuần tự (Random Proj → Analytic → Desc → Contrastive Adapter) | Đo retrieval accuracy trên cả old & new relations; không replay instance cũ. |
| **M7.6** | Parameter-Matched Comparison & Multi-Seed (`2021..2025`) | Colab Multi-Seed (sau khi chốt config bằng validation split) | Có mean ± std, kiểm soát task-order sensitivity, cập nhật registry & luận văn. |

---

## 2. Quy tắc Validation Split & Chống Test-Set Overfitting

- Seed `2021` được xem là **development evidence** (không dùng final test metrics để chọn hyperparameters $\eta, \gamma, \lambda, K$).
- Trước khi tune $\eta, \gamma, \lambda, K$ ở M7.3–M7.4, cần tách validation subset từ `train_samples` (hoặc dùng nested validation split theo task) và ghi nhận bằng Research Decision Record.

---

## 3. Phân tích tác động mã nguồn (Implementation Impact Analysis)

- **Không sửa:** `src/models/*`, `experiments/run_p*.py`, `configs/dynatree_cre/dt_c0..c4_*`, `runs/dynatree_cre_colab_results/`.
- **Mở rộng có kiểm soát trong `src/dynatree_cre/`:**
  - `prototype_bank.py`: Hỗ trợ trả về cả `raw_prototype` và `transformed_prototype` cho Dual-Space Routing (M7.2).
  - `router.py`: Thêm `size_corrected_hierarchical`, `hybrid_tree_routing` (`gamma`, `eta`), và chế độ trả về candidate prompt paths cho reranking (M7.3–M7.4).
  - `metrics.py` & `trainer.py`: Bổ sung per-leaf gradient tracking, `Recall@1/3/5/10`, 2×2 routing decomposition, và candidate reranking forward pass (batched).
  - `experiments/`: Thêm script diagnostic đọc `.pt` trên Colab (`diagnose_dynatree_checkpoint.py`) và giữ nguyên `run_dynatree_cre.py` tương thích ngược.
