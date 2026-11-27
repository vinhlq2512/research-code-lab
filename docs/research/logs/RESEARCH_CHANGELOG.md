# Research Changelog (DynaTree-CRE)

## 2026-11-27 — Research Direction Migration: TAPTA → DynaTree-CRE

### Context
Sau khi hoàn thành chuỗi thực nghiệm TAPTA (`E007`–`E018`) và phân tích lỗi `E016`–`E017`, phương pháp tổ chức prompt theo Task ID đạt trần $58.46 \pm 0.45\%$ (FewRel) và $42.91 \pm 0.51\%$ (TACRED), trong khi bản chỉnh sửa `P5` không cải thiện được hiệu năng (`56.49%` FewRel).

### Previous Approach
**TAPTA (Continual Relation Extraction with Task-Aware Prompt Adaptation):** Cây 2 tầng $P^{\text{root}} + P^{\text{leaf}}_t$ gắn theo Task ID, định tuyến theo Task Centroid + Sub-Centroids và Decision-Level Soft Fusion (`P4`).

### New Approach
**DynaTree-CRE (Working Title):** Dynamic Semantic Prompt Trees for Rehearsal-Free Continual Relation Extraction with Task-Agnostic Routing. Cây phân cấp động `Root → Semantic Cluster → Relation`, hình thành tuần tự từ prototype ngữ nghĩa của các quan hệ đã quan sát (tâm hóa tại $T_1$ để kháng anisotropy), chia sẻ cluster prompt xuyên tác vụ, định tuyến mềm phân cấp không cần Task ID.

### Reason for Change
Task split trong CRE phân tán các quan hệ cùng ngữ nghĩa qua nhiều task khác nhau, khiến prompt theo task bị pha loãng ngữ nghĩa, không chia sẻ được tri thức xuyên tác vụ và làm centroid các task chồng lấn (`cos = 0.958`).

### Evidence
- `E012`, `E013`, `E016`, `E017`, `E018` trong `experiments/registry.yaml`.
- Phân bố quan hệ thực tế trong `ProtocolV1Dataset(seed=2021)`.

### Affected Documents
- Đánh dấu `superseded`: `docs/specifications/architecture-freeze-memo.md`, `docs/research/p4-diagnosis/P5_SPECIFICATION.md`.
- Đánh dấu `archived-comparator-spec`: `docs/specifications/p4-hierarchical-prompt-tree-spec.md`.
- Tạo mới: `docs/research/migrations/RESEARCH_DIRECTION_MIGRATION.md`, `RESEARCH_DIRECTION.md`, `RELATED_WORK.md`, `decisions/RD-001`..`RD-012`, `methods/dynatree-cre/*`, `experiments/EXPERIMENT_PROTOCOL.md`.
- Cập nhật trỏ hướng active: `AGENTS.md`, `README.md`, `workflow/STATUS.md`, `workflow/DECISIONS.md`.

### Affected Code
- Bảo tồn nguyên vẹn: `src/models/*`, `experiments/run_p*.py`, `experiments/continual-relation-baselines/*`, `runs/*`, `results/*`.
- Bổ sung mới: `src/dynatree_cre/*`, `tests/dynatree_cre/*`, `configs/dynatree_cre/*`, `experiments/run_dynatree_cre.py`.

### Affected Experiments
- Giữ nguyên toàn bộ ID `E001`–`E018` làm lịch sử đối chứng.
- Dùng prefix mới `DT-C0`..`DT-C6` cho DynaTree-CRE.

### Risks
Anisotropy của BERT đóng băng, bẫy zero-grad khi Top-K routing lúc train, negative transfer trong cụm chưa chuẩn xác, và chi phí compute full 8-task cần Colab GPU (Gate 5).

### Next Actions
Hoàn thành triển khai M1–M6, chạy full unit/invariant/integration test suite và các smoke runs cục bộ dưới 2 phút trên MPS.
