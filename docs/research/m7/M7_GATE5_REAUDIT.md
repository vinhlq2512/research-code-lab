# M7.0 — Gate 5 Re-Audit Report (`DynaTree-CRE` MVP)

**Ngày audit:** 2026-11-28  
**Commit thực thi Colab:** `8fc4c81` (nhánh `feat/dynatree-cre`)  
**Nguồn dữ liệu kiểm toán:** `runs/dynatree_cre_colab_results/` (giải nén từ `dynatree_cre_mvp_colab_results.zip`) và source code `src/dynatree_cre/`.

---

## 1. Xác minh số lượng & tính hợp lệ các lượt chạy (13 runs)

Trong bundle Colab có **13** thư mục có `metrics.json`:
- **12 lượt chạy hợp lệ & khác biệt về mục đích:**
  1. `dt_c0_prototype_control/fewrel_5shot/seed_2021` (`t1_centered_cosine`, FewRel, 8 tasks, `dfp=105e7ede38`)
  2. `dt_c0_raw_cosine/fewrel_5shot/seed_2021` (`raw_cosine`, FewRel, 8 tasks, `dfp=105e7ede38`)
  3. `dt_c1_flat_hard/fewrel_5shot/seed_2021` (FewRel, 8 tasks)
  4. `dt_c2_flat_soft/fewrel_5shot/seed_2021` (FewRel, 8 tasks)
  5. `dt_c3_cluster_flat_route/fewrel_5shot/seed_2021` (FewRel, 8 tasks)
  6. `dt_c4_hier_soft/fewrel_5shot/seed_2021` (FewRel, 8 tasks)
  7. `dt_c4_resume_check/fewrel_5shot/seed_2021` (FewRel resume `after_T4 -> T8`)
  8. `dt_c4_hier_soft/tacred_5shot/seed_2021` (TACRED, 8 tasks, 41 relations, `dfp=96400b4de5`)
  9. `dt_c2_semantic_homogeneous/fewrel_5shot/seed_2021` (Exp B, `dfp=56f3aed1d6`)
  10. `dt_c4_semantic_homogeneous/fewrel_5shot/seed_2021` (Exp B, `dfp=56f3aed1d6`)
  11. `dt_c2_cross_task_dispersed/fewrel_5shot/seed_2021` (Exp C, `dfp=9531341f50`)
  12. `dt_c4_cross_task_dispersed/fewrel_5shot/seed_2021` (Exp C, `dfp=9531341f50`)
- **1 lượt chạy bị trùng / gắn nhãn sai (Mislabeled Duplicate):**
  - `dt_c0_prototype_control/tacred_5shot/seed_2021`: `metrics.json` và `manifest.json` ghi nhận `dataset: fewrel`, `dfp=105e7ede38`, `num_leaves=80`, `NCM Acc=58.61%` — trùng hoàn toàn với `dt_c0_prototype_control/fewrel_5shot/seed_2021`. Nguyên nhân: Cell 6 của notebook `8fc4c81` gọi nhầm `--config configs/dynatree_cre/dt_c0_prototype_control_fewrel_5shot_seed2021.yaml`.
  - **Khắc phục:** Đã tạo `configs/dynatree_cre/dt_c0_prototype_control_tacred_5shot_seed2021.yaml` và cập nhật Cell 6 notebook để chạy cả `DT-C0 (t1_centered_cosine)` và `DT-C0 (raw_cosine)` trên TACRED ở lượt Colab tiếp theo.

---

## 2. Audit Chi tiết Source Code & Metric Definitions

### 2.1. Oracle Accuracy (`trainer.py:384-393`, `router.py:99-111`)
- **Thông tin đặc quyền (Privileged Info):** Sử dụng **`gold_relation_ids` (Relation ID)** ở test time (`strategy_override="oracle_route"`), **không** dùng Task ID hay Cluster ID trực tiếp.
- **Candidate space & Prompt:** Chọn duy nhất 1 leaf prompt ứng với `gold_relation_id` (`topk_weights = 1.0`), ghép `P_root + P_cluster(gold) + residual_P_gold` (hoặc flat leaf prompt nếu không có root/cluster).
- **Classifier sử dụng:** `ores["pred_cos_rels"]` — tức **`ExpandableCosineClassifier`** (không tính Oracle cho `PromptedNCM`).
- **Hệ quả quan trọng:** Không được coi Oracle Accuracy là mức accuracy chắc chắn đạt được nếu cải thiện routing, vì (a) `DT-C2/C3/C4` train bằng soft Top-3 mixture nhưng `oracle_route` eval bằng $K=1$ pure prompt (train-eval mismatch), và (b) `ExpandableCosineClassifier` vẫn chịu new-class bias.

### 2.2. Zero-Gradient Check (`trainer.py:262-268`)
- Code hiện tại kiểm tra:
  ```python
  has_leaf_grad = any(
      self.model.prompt_bank.leaf_prompts[lid].grad is not None
      and float(self.model.prompt_bank.leaf_prompts[lid].grad.abs().sum().item()) > 0.0
      for lid in current_leaf_ids
  )
  ```
- **Lỗ hổng phát hiện:** Hàm dùng `any(...)` trên toàn bộ `current_leaf_ids` của batch chứ **chưa theo dõi từng trainable prompt riêng lẻ** (một leaf trong batch có gradient là `zero_prompt_grad_steps` không tăng, kể cả khi có một leaf khác trong task không bao giờ nhận gradient). Cần bổ sung per-leaf gradient tracking trong M7.

### 2.3. PromptedNCM Prototype Construction (`trainer.py:122-158, 285`)
- `_compute_prompted_class_means(task, use_effective_leaf_prompt=False)` gọi `self.model.forward_batch(..., gold_relation_ids=None)`:
  - Prototype của `PromptedNCM` được tạo bằng **predicted routing** (nhất quán với lúc inference), **không** dùng gold routing.
  - Query và prototype đều dùng cùng biểu diễn đầu ra `out["representations"]` của `FrozenBERT` sau khi đi qua `composed_prompt`, rồi chuẩn hóa L2 và lấy trung bình theo class (`classifier.py:111-114`).
  - **Không** tái tạo old prototypes từ dữ liệu cũ; chỉ append prototype của task mới.

### 2.4. Hierarchical Router (`router.py:71-88, 117-121`)
- `hierarchical_soft` tính `log p(c|x)` trên toàn bộ `C` cụm và `log p(r|c,x)` trong từng cụm, sau đó lấy `topk(rel_probs, k=3)` trên toàn bộ `R_seen` leaves.
- Đây là **global hierarchical soft routing** (không phải hard cluster routing chỉ xét 1 cụm).
- **Lỗi cấu trúc toán học:** `log_cond = F.log_softmax(leaf_sims[:, member_positions] / tau_r, dim=-1)` chia chuẩn hóa theo `cluster_size`, tạo ra **cluster-size bias** bất lợi cho cụm lớn (xem `ROUTING_ERROR_ANALYSIS.md`).

### 2.5. Checkpoint Resume Verification
- Trong `dynatree_cre_mvp_colab_results.zip`, các file `*.pt` đã bị loại trừ để giảm dung lượng tải về, nên trên máy local mới xác minh được **metric equality** và **tree JSON equality** giữa `dt_c4_hier_soft` và `dt_c4_resume_check`.
- **Chưa tuyên bố bit-exact tensor equality & prediction equality trên Colab** cho tới khi chạy kiểm tra trực tiếp trên các file `.pt` lưu trên Google Drive (`MyDrive/TAPTA_DynaTree_CRE/runs/dynatree_cre`).
