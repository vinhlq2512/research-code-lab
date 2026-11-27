# Giao Thức Thực Nghiệm: DynaTree-CRE (Experiment Protocol)

## 1. Bảo Toàn Protocol v1 (Source of Truth)

DynaTree-CRE kế thừa hoàn toàn **Protocol v1** (`docs/protocol-v1.md`) để đảm bảo so sánh công bằng với toàn bộ các thí nghiệm lịch sử (`B0`–`B5`, `P0`–`P4`):

| Tham số | FewRel Track A (Primary) | TACRED Track A (Secondary) | Ghi chú |
|---|---|---|---|
| **Dataset source** | `data/protocol_v1/fewrel/CFRLdata_10_100_10_5` | `data/protocol_v1/tacred/CFRLdata_10_100_10_5` | Không thay đổi split hay lọc mẫu riêng |
| **Tasks × Rels/Task** | `8 × 10 = 80` relations | `8 × 5 = 40` relations *(chú ý discrepacy)* | `relation_name.txt` của TACRED có **40** dòng; config cũ ghi `41` |
| **Train shots** | Strict **5-shot** (kể cả T1 = 50 mẫu) | Strict **5-shot** (kể cả T1 = 25 mẫu) | Tổng 400 (FewRel) / 200 (TACRED) mẫu train |
| **Test samples** | 100/relation = 8,000 mẫu | Tới 100/relation (~1,550–2,122 mẫu) | Đánh giá tích lũy toàn bộ seen relations |
| **Backbone** | `bert-base-uncased` (100% Frozen) | `bert-base-uncased` (100% Frozen) | Entity markers `[E1]`, `[/E1]`, `[E2]`, `[/E2]` |
| **Prompt token budget** | $L = 8$ tokens (default, matching P0–P4) | $L = 8$ tokens | $L = 10$ có thể kiểm chứng qua ablation |
| **Replay exemplars** | `0` (`M=0`, no raw/feature/gen replay) | `0` | Nghiêm cấm lưu instance cache |
| **Task ID at inference** | **Unavailable** (Task-Agnostic CIL) | **Unavailable** | Oracle chỉ tách làm control riêng |
| **Debug seed** | `2021` | `2021` | Multi-seed: `[2021, 2022, 2023, 2024, 2025]` |
| **Compute Environment** | **Google Colab GPU (`device: cuda`)** | **Google Colab GPU (`device: cuda`)** | **Bắt buộc Colab-Only** cho 100% real-data runs (kể cả smoke & debug); không chạy trên MacBook local |

---

## 2. Bảng Các Biến Thể Thử Nghiệm (`DT-C0` đến `DT-C6`)

Sử dụng prefix `DT-` để **không xung đột** với bảng `B0–B5` và `P0–P5` trong `experiments/registry.yaml`:

| ID | Variant Name | Query Space | Prompt Structure | Routing Strategy | Prompt Budget ($L$) | Trạng thái |
|---|---|---|---|---|---|---|
| **DT-C0** | Frozen Encoder + Prototype Control | `t1_centered_cosine` (và `raw_cosine` control) | None | Nearest Prototype (no prompt) | 0 | `MVP` |
| **DT-C1** | Flat Relation Prompt + Hard Routing | `t1_centered_cosine` | Leaf only ($\Delta P_r$) | `flat_hard` (Top-1 leaf) | 8 | `MVP` |
| **DT-C2** | Flat Relation Prompt + Soft Routing | `t1_centered_cosine` | Leaf only ($\Delta P_r$) | `flat_soft_topk` ($K=3, \tau=0.2$) | 8 | `MVP` |
| **DT-C3** | Incremental Cluster Prompt + Flat Leaf Routing | `t1_centered_cosine` | Tree ($P_{\text{root}} + P_v + \Delta P_r$) | Flat Top-K trên leaf prototypes | 8 | `MVP` |
| **DT-C4** | **DynaTree-CRE (Main Candidate)** | `t1_centered_cosine` | Tree ($P_{\text{root}} + P_v + \Delta P_r$) | `hierarchical_soft` ($\tau_c=0.2, \tau_r=0.2, K=3$) | 8 | `MVP` |
| **DT-C5** | DT-C4 + Controlled Shared Prompt Adaptation | `t1_centered_cosine` | Tree + adaptive cluster prompt | `hierarchical_soft` | 8 | `POST-MVP (M7)` |
| **DT-C6** | DT-C4 + Additional Knowledge Consolidation | `t1_centered_cosine` | Tree + KD/Contrastive | `hierarchical_soft` | 8 | `POST-MVP (M7)` |

---

## 3. Phân Biệt Hai Hệ Evaluator Trong Repository (Audit Discrepancy)

1. `experiments/run_sequential_ft.py` (`E001–E004`) dùng loader trong `dataset-pipelines/continual-relation-extraction/` (tập test `val_wiki.json` 55,200 mẫu).
2. `experiments/run_p0_p1.py`..`run_p5.py` (`E007–E018`) dùng `ProtocolV1Dataset` (`data/protocol_v1/`, `test_0.txt` 8,000 mẫu cho FewRel).
3. DynaTree-CRE sử dụng **chính xác `ProtocolV1Dataset`** để so sánh trực tiếp, công bằng 1-1 với `P0–P4` và `D09` (`59.93%`).

---

## 4. Ba Phân Phối Tác Vụ (Experiments A, B, C)

- **Experiment A (`standard_random`):** Thứ tự chuẩn `conpl_seed_shuffle` của Protocol v1 (Main Benchmark).
- **Experiment B (`semantic_homogeneous`):** Gom các quan hệ gần nghĩa vào cùng một task (dùng agglomerative clustering trên prior embeddings `rel_feature.npy` offline trước khi chạy benchmark; **tuyệt đối không truyền nhãn cụm hay thông tin tương lai cho mô hình**). Kiểm tra khả năng tạo cluster mới.
- **Experiment C (`cross_task_dispersed`):** Phân tán đều các quan hệ cùng cụm ngữ nghĩa ra nhiều task khác nhau (round-robin từ các cụm trong `rel_feature.npy`). Stress-test khả năng **cross-task knowledge sharing** của `DT-C4` so với `DT-C2`.

---

## 5. Bộ Chỉ Số Bắt Buộc

1. **CL Accuracy & Forgetting (cho cả `cosine_head` và `ncm_head`):** `Final AA`, `AIA`, `Average Forgetting`, `BWT`, `Old Relations Acc`, `New Relations Acc`, `Performance Matrix` $A_{t,j}$.
2. **Routing Metrics:** `Relation Routing Top-1 Acc`, `Relation Routing Recall@K`, `Cluster Routing Acc`, `Mean Routing Entropy`, `Cross-task Routing Confusion`, `Oracle Routing Gap`.
3. **Confusion Breakdown:** `Old-to-New`, `New-to-Old`, `Within-Cluster`, `Cross-Cluster`, `Cross-Task`.
4. **Tree Metrics:** `num_nodes`, `num_clusters`, `num_leaves`, `max_depth`, `cluster_size_min/mean/max`, `singleton_cluster_ratio`, `cluster_cohesion`, `cluster_separation`, `tree_growth_per_task`, `old_path_retention`, `cluster_prototype_movement`.
5. **Efficiency Metrics:** `persistent_prototype_bytes`, `persistent_prompt_bytes`, `classifier_bytes`, `trainable_param_count`, `total_persistent_param_count`, `checkpoint_bytes`, `train_time_sec`, `eval_time_sec`, `inference_latency_ms_per_sample`.
