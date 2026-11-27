# Experiment Log (DynaTree-CRE)

> **Quy tắc hạ tầng thực thi bắt buộc (Colab-Only Policy):** **100% các lượt chạy thực nghiệm chính thức** (bao gồm smoke runs, baselines, ablations, debug 1 seed, stress tests Exp B/C và multi-seed) **phải chạy trên Google Colab GPU (`device: cuda`)** qua [`notebooks/dynatree_cre_mvp_colab_seed2021.ipynb`](../../../notebooks/dynatree_cre_mvp_colab_seed2021.ipynb), không chạy thực nghiệm trên máy MacBook local. Các lượt chạy MPS bên dưới chỉ được ghi nhận là **`LOCAL_CODE_SANITY_ONLY`** trước khi khóa quy tắc này; mọi số liệu chính thức sẽ được cập nhật từ kết quả Colab GPU.

> Quy tắc: Mọi kết quả thực nghiệm bên dưới đều xuất phát trực tiếp từ các lệnh đã chạy trên phần cứng thật. Các thí nghiệm chưa chạy trên Colab GPU ghi rõ `PENDING_COLAB_GPU`.

---

## DT-E001 — `DT-C0` Control Comparison (`t1_centered_cosine` vs `raw_cosine`, FewRel 8 Tasks, Seed 2021)

- **Research Question:** RQ1, RQ5, H1 (Anisotropy vs stationary prototype space).
- **Dataset & Task Order:** FewRel Track A (`8 tasks × 10 rels`, 5-shot train, 400 test samples/task = 3,200 test samples ở T8, seed 2021).
- **Device:** Apple Silicon MPS (`bert-base-uncased` frozen 109.5M params).

| Run ID | Similarity Space | Threshold Policy | Final AA (NCM) | AIA | Forgetting | Clusters (T8) | Singletons Ratio | Cluster Separation | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `dt_c0_t1_centered_smoke` | `t1_centered_cosine` | `t1_quantile=0.70` | **58.13%** | 68.31% | 12.75% | **4** (`[12..26]`) | **0.0%** | **0.932** | `LOCAL_CODE_SANITY_ONLY` |
| `dt_c0_raw_cosine_smoke` | `raw_cosine` | `fixed=0.75` | **59.41%** | 69.88% | 12.11% | **17** (`[1..30]`) | **64.7%** | **0.373** | `LOCAL_CODE_SANITY_ONLY` |

- **Interpretation & Semantic Tree Evidence:**
  - `raw_cosine` với ngưỡng `0.75` bị sụp đổ phân cụm do hình nón BERT (11/17 cụm là singleton, 1 cụm khổng lồ 30 relations).
  - `t1_centered_cosine` tách sạch thành 4 cụm cân đối (0% singleton). Kiểm tra `trees/task_08.json` cho thấy `cluster_003` (`n=12`) tự động gom được **`sibling(T1)`, `child(T1)`, `spouse(T3)`, `father(T4)`, `mother(T4)`** từ 4 task khác nhau mà không dùng Task ID.
  - Tuy nhiên, trên tác vụ nearest-prototype thuần túy không train prompt, `raw_cosine` vẫn nhỉnh hơn `t1_centered_cosine` `+1.28%` (`59.41%` vs `58.13%`), cho thấy bước trừ tâm cố định từ $T_1$ hơi làm thay đổi góc phân loại cho các task sau (cần theo dõi trong H1).

---

## DT-E002 — `DT-C4` Real-Data T1–T2 Full Test & Checkpoint Resume Verification

- **Research Question:** H2, H4, H5, Kiểm chứng End-to-End MVP & Bit-exact Resume.
- **Dataset:** FewRel Track A (`T1–T2`, 20 relations, 5-shot train = 100 mẫu, full 1,000 test/task = 2,000 test samples, 15 epochs, $L=8$, seed 2021).
- **Commands:**
  - Full T1–T2: `.venv/bin/python experiments/run_dynatree_cre.py --config configs/dynatree_cre/dt_c4_dynatree_hier_soft_fewrel_5shot_seed2021.yaml --device mps --max-tasks 2 --output-dir runs/dynatree_cre/dt_c4_t1_t2_smoke/fewrel_5shot/seed_2021`
  - Resume (`after_T1 -> T2`): `.venv/bin/python experiments/run_dynatree_cre.py ... --resume-from runs/dynatree_cre/dt_c4_t1_t2_smoke/fewrel_5shot/seed_2021/checkpoints/after_T1/state.pt --output-dir runs/dynatree_cre/dt_c4_t1_t2_resume_check/fewrel_5shot/seed_2021`

| Run | Cosine Acc (T2) | Cosine F | NCM Acc (T2) | NCM F | Oracle Acc | Routing Top-1 | Recall@3 | Cluster Top-1 | Zero-Grad Steps | Persistent Mem | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dt_c4_t1_t2_smoke` | **72.50%** | 21.30% | **76.65%** | **6.80%** | 94.95% | 74.00% | 90.75% | 86.35% | **0** | 1,112,064 B (~1.06 MB) | `LOCAL_CODE_SANITY_ONLY` |
| `dt_c4_t1_t2_resume_check` | **72.50%** | 21.30% | **76.65%** | **6.80%** | 94.95% | 74.00% | 90.75% | 86.35% | **0** | 1,112,064 B (~1.06 MB) | `VERIFIED_EXACT (Local Sanity)` |

- **Interpretation:**
  - Resume khôi phục chính xác 100% số liệu, 4 clusters, 20 leaves (zero duplicate nodes).
  - Không xảy ra bẫy zero-gradient ở `T2` (`target_in_topk_rate = 100%`, `zero_prompt_grad_steps = 0`).
  - Lộ rõ **new-class bias** trên `ExpandableCosineClassifier` (`old_acc = 65.9%` vs `new_acc = 79.1%`, forgetting `21.30%`) so với `PromptedNCM` (`old_acc = 79.2%` vs `new_acc = 74.1%`, forgetting `6.80%`), xác nhận giá trị thiết kế dual head (`RD-010`).

---

## DT-E003 — `DT-C1` to `DT-C4` Apples-to-Apples T1–T2 Ablation Smoke Matrix

- **Setting:** FewRel Track A (`T1–T2`, 200 test samples/task = 400 test samples ở T2, 5 epochs, $L=8$, seed 2021, MPS).

| Run ID | Variant | Cosine Acc | Cosine F | NCM Acc | NCM F | Oracle Acc | Routing Top-1 | Recall@K | Cluster Top-1 | Zero-Grad | Mem (Bytes) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dt_c1_smoke_cmp` | `DT-C1` (Flat Hard, K=1) | **76.75%** | **9.50%** | 75.50% | 9.00% | 95.75% | 74.75% | 74.75% | 87.50% | 0 | 989,184 | `LOCAL_CODE_SANITY_ONLY` |
| `dt_c2_smoke_cmp` | `DT-C2` (Flat Soft, K=3) | 76.00% | 19.00% | 76.25% | 11.50% | 92.50% | 74.75% | 94.75% | 87.50% | 0 | 989,184 | `LOCAL_CODE_SANITY_ONLY` |
| `dt_c3_smoke_cmp` | `DT-C3` (Tree + Flat Route) | 74.75% | 20.00% | **78.00%** | 11.00% | 93.00% | 74.75% | 94.75% | 87.50% | 0 | 1,112,064 | `LOCAL_CODE_SANITY_ONLY` |
| `dt_c4_smoke_cmp` | `DT-C4` (Tree + Hier Route) | 75.25% | 14.50% | 77.25% | **4.00%** | 92.00% | 71.75% | 90.25% | 83.50% | 0 | 1,112,064 | `LOCAL_CODE_SANITY_ONLY` |

- **Interpretation:** Trên smoke 2 tasks, `DT-C4` có forgetting thấp nhất ở NCM head (`4.00%`), còn `DT-C3` đạt NCM accuracy cao nhất (`78.00%`) nhờ giữ Recall@3 `94.75%` của flat routing trên cây prompt (phù hợp với giả thuyết H7 cần kiểm tra ở full 8 tasks). Không được kết luận ưu thế phương pháp chỉ từ smoke 2 tasks.

---

## Pending Official Experiments (Colab-Only — Require Gate 5 Approval)

| Experiment ID | Variant & Scope | Status | Artifact / Notebook |
|---|---|---|---|
| `DT-E004` | `DT-C0..DT-C4` Full 8-Task FewRel Track A (Seed 2021) | `PENDING_COLAB_GPU` | `notebooks/dynatree_cre_mvp_colab_seed2021.ipynb` |
| `DT-E005` | `DT-C4` Full 8-Task TACRED Track A (40 rels, Seed 2021) | `PENDING_COLAB_GPU` | `configs/dynatree_cre/dt_c4_dynatree_hier_soft_tacred_5shot_seed2021.yaml` |
| `DT-E006` | `DT-C2` vs `DT-C4` Exp B (`semantic_homogeneous`) & Exp C (`cross_task_dispersed`) | `PENDING_COLAB_GPU` | `notebooks/dynatree_cre_mvp_colab_seed2021.ipynb` (`Part C`) |
| `DT-E007` | Multi-Seed Verification (`2021..2025`) trên FewRel & TACRED | `PENDING_COLAB_GPU` | Pending `DT-E004` & `DT-E005` |
