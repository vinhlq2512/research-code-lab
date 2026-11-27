# Implementation Log (DynaTree-CRE)

## 2026-11-27 — M0 Migration & Specification Setup

### Objective
Thiết lập nhánh `feat/dynatree-cre`, kiểm toán toàn bộ repo `research-code-lab`, đánh dấu tài liệu cũ không phá hủy và hoàn thiện bộ đặc tả M0–M6 + RDRs (`RD-001`..`RD-012`).

### Completed Work
- Kiểm toán working tree sạch tại commit `330fc0f` (`feat/tapta-data-scalability`).
- Xác nhận 74/74 unit tests hiện có pass trong `.venv`.
- Tạo nhánh mới `feat/dynatree-cre` và cập nhật Source-of-Truth pointers (`AGENTS.md`, `README.md`, `workflow/STATUS.md`, `workflow/DECISIONS.md`) cùng Obsidian Vault.

---

## 2026-11-27 — M1–M6 Implementation, Bug Fixes & Verification

### Objective
Triển khai đầy đủ kiến trúc MVP của DynaTree-CRE (`src/dynatree_cre/*`), 6 cấu hình chuẩn (`configs/dynatree_cre/*.yaml`), CLI runner, Colab notebook, bộ kiểm thử TDD (`tests/dynatree_cre/*`) và chạy xác minh trên dữ liệu thật dưới 2 phút trên MPS.

### Completed Work
- **M1:** `StableQueryEncoder` (`eval()` mode cố định, resize marker trước khi freeze, hỗ trợ `entity_pair`/`cls`/`mean_pool` và `synthetic_tiny` cho test), `SemanticPrototypeBank` (`t1_centered_cosine`, `incremental_centered_cosine`, `raw_cosine`, kiểm tra zero/NaN/Inf).
- **M2–M5:** `DynamicSemanticTree` (average-linkage HAC tại T1, chặn Ward+cosine, chèn gia tăng deterministic theo `relation_id`, kiểm tra 11 invariants, JSON round-trip & ASCII tree), `HierarchicalPromptBank` ($L=8$, residual cộng tính, freeze kép `requires_grad=False` + loại khỏi `AdamW`), `TaskAgnosticRouter` (log-space hierarchical soft routing, `flat_hard`, `flat_soft_topk`, `dense_soft`, isolated `oracle_route`), `ExpandableCosineClassifier` + `PromptedNCMClassifier` song song.
- **M6:** `DynaTreeCREModel`, `DynaTreeContinualTrainer` (7 pha A–G, kiểm tra `torch.equal` snapshot tensor đóng băng sau mỗi task), `checkpoint.py` (kèm RNG CPU + MPS), `benchmark.py` (Protocol v1 + Exp B/C constructors + synthetic benchmark), `experiments/run_dynatree_cre.py`, `notebooks/dynatree_cre_mvp_colab_seed2021.ipynb`.

### Issues Encountered & Fixed
1. **`pytest.ini` namespace shadowing (`src`):** Đặt `tests/dynatree_cre` đầu `testpaths` trong `pytest.ini` để `REPO_ROOT` được thêm vào `sys.path` trước khi thu thập `experiments/tests/test_p5_modules.py`.
2. **Flat prompt device mismatch trên MPS:** Khi `use_root_prompt=False` và `use_cluster_prompts=False` (`DT-C1`/`DT-C2`), `HierarchicalPromptBank.synchronize_with_tree` bị rơi về CPU. Đã sửa bằng `_device_anchor` buffer + fallback `word_embedding_weight.device` và bổ sung test `test_flat_prompt_bank_without_root_inherits_target_device`.
3. **MPS RNG resume determinism:** Đã bổ sung `torch.mps.get_rng_state()` / `set_rng_state()` trong `checkpoint.py` và `torch.mps.manual_seed()` trong `trainer.py`, đưa resume `DT-C4` (`after_T1 -> T2`) về **khớp chính xác 100%** mọi chữ số (`0.7250` cosine / `0.7665` NCM).

### Tests Executed & Results
- `.venv/bin/pytest -q`: **93 passed in 13.11s** (74 existing baseline/pipeline tests + 19 DynaTree-CRE tests).

### Known Limitations
- Full 8-task training runs (`DT-C1..DT-C4`) trên FewRel/TACRED và 5-seed evaluation chưa chạy vì giới hạn compute local (< 2 phút/lệnh, cần Google Colab GPU theo Gate 5).
