# Báo Cáo Chuyển Hướng Nghiên Cứu (Research Direction Migration Report)

- **Ngày lập:** 2026-11-27
- **Repository:** `/Users/vinhlq2512/code/research-code-lab` (nhánh `feat/dynatree-cre`, tách từ `feat/tapta-data-scalability` tại commit `330fc0f`)
- **Obsidian Vault:** `/Users/vinhlq2512/Obisidian/Knowledge Hub`
- **Đề tài Luận văn Thạc sĩ (Project gốc, giữ nguyên):** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPA-CRE / TAPTA)
- **Giai đoạn phương pháp cũ (trong luận văn TAPTA):** Task-Level Prompt Tree & Adaptation (`P0–P5`, `E007–E018`)
- **Phương pháp mục tiêu mới cho luận văn TAPTA (Working Title):** **DynaTree-CRE: Dynamic Semantic Prompt Trees for Rehearsal-Free Continual Relation Extraction with Task-Agnostic Routing**

---

## 1. Tóm Tắt Hướng Nghiên Cứu Trước Đây (TAPTA)

### 1.1. Mục tiêu trước đây
Hướng nghiên cứu ban đầu (**TAPTA**) giải bài toán Continual Relation Extraction (CRE) theo cách tiếp cận **Task-Aware Prompt Adaptation**:
1. Giữ cố định pretrained backbone (`bert-base-uncased`, 109.5M tham số) để tránh drift ở tầng biểu diễn nền.
2. Tổ chức prompt theo **Task ID**: mỗi task $T_t$ sở hữu một nhánh prompt riêng ($P^{\text{leaf}}_t$ ghép với $P^{\text{root}}$ cố định sau $T_0$, tổng $L = 4 + 4 = 8$ tokens).
3. Sử dụng bộ định tuyến ở mức task (`TaskRouter` dùng Centroid + Relation Sub-Centroids với $\alpha = 0.7$, $K = 2$, $\tau = 0.2$) và tổ hợp ở tầng quyết định (Decision-Level Soft Fusion) để suy luận không cần Oracle Task ID.

### 1.2. Các thành phần đã hoàn thành ở hướng cũ
Qua kiểm toán thực tế `research-code-lab` (`commit 330fc0f`) và Obsidian Vault:
- **Pipeline dữ liệu & đánh giá:**
  - `dataset-pipelines/continual-relation-extraction/`: loader chuẩn, `PerformanceMatrix`, `compute_continual_summary_metrics`, test suite 62 unit tests.
  - `src/data/protocol_v1_loader.py` & `src/data/scalable_fewrel_loader.py`: chuẩn **Protocol v1 Track A** (FewRel 8×10 5-shot, TACRED 8×5 5-shot, entity markers `[E1]`, `[/E1]`, `[E2]`, `[/E2]`, lọc trùng hash test toàn cục).
- **Baselines đã thực thi thật trên phần cứng:**
  - `E001` (`B0` Sequential FT, `dataset-pipelines` 55,200 test samples, MPS): $A_{\text{last}} = 8.76\%$, $F_T = 33.62\%$.
  - `E002` (Upper Bound Joint Training, MPS): Overall Test Acc $= 52.12\%$.
  - `E003` (Frozen BERT + Linear Head, MPS): $A_{\text{last}} = 10.33\%$, $F_T = 86.27\%$.
  - `E004` (Frozen BERT + Nearest Prototype NCM, MPS): $A_{\text{last}} = 59.63\%$, AIA $= 70.63\%$, $F_T = 10.45\%$.
  - `E006` (`B5` WAVE-CRE smoke 5/10 tasks theo Track B authors' setting, Colab GPU): 5-task Acc $= 91.68\%$, $F = 3.50\%$ (bị dừng ở task 5 do OOM RAM 51 GB trên T4 với 1.6M latent replay dicts).
- **Chuỗi TAPTA đã chạy thật (`ProtocolV1Dataset`, FewRel 8,000 test samples / TACRED 2,122 test samples, Colab CUDA & MPS):**
  - `E007` (`P0` Shared Prompt): Proto Acc $= 52.83\%$, Linear CIL $= 9.84\%$.
  - `E008` (`P1` Oracle-Task Prompt): Proto Acc $= 70.14\%$, TIL $= 82.75\%$, Linear CIL $= 9.84\%$.
  - `E009` (`P2a` Centroid Cosine Router): Routed-Gated $= 59.35\%$, Proto $= 59.28\%$, Router Top-1 $= 64.69\%$, AIA $= 69.71\%$, $F = 11.91\%$.
  - `E010` (`P2b` Learned MLP Router + Gaussian latent feature replay): Proto $= 56.04\%$, Routed-Gated $= 46.25\%$, Router Top-1 $= 50.85\%$.
  - `E011` (`P3` Soft Top-K Prompt Blend): Proto $= 55.50\%$, Top-1 Gated $= 59.25\%$, Top-2 Coverage $= 81.41\%$.
  - `E012` (`P4` FewRel, seed 2021): Decision Fusion $= 57.58\%$, Tree Top-1 Proto $= 58.38\%$, Tree Top-1 Gated $= 58.80\%$, AIA $= 68.86\%$, $F = 11.89\%$.
  - `E013` (`P4` TACRED, seed 2021): Decision Fusion $= 42.45\%$, Tree Top-1 Proto $= 41.85\%$, AIA $= 63.46\%$, $F = 14.55\%$.
  - `E016` (Chẩn đoán D01–D09): Xác định 4 nguyên nhân gốc rễ (Anisotropy `cos = 0.958` FewRel / `0.933` TACRED; Margin Collapse `0.010–0.023`; Toxic Soft Decision Fusion; Root Freeze Plasticity Loss) và ghi nhận `D09` Unprompted Frozen BERT Flat Prototype $= 59.93\%$ (FewRel) / $41.52\%$ (TACRED).
  - `E017` (`P5` candidate ablation): `P5-Full` đạt $56.49\%$ trên FewRel (thấp hơn `P4` $57.58\%$) và `P5-B` đạt $41.50\%$ trên TACRED (thấp hơn `P4` $42.45\%$) $\rightarrow$ giả thuyết P5 chưa cải thiện trên P4.
  - `E018` (`P4` Multi-Seed 5 seeds 2021–2025): FewRel Decision Fusion $57.74 \pm 0.36\%$, Hard Top-1 $58.46 \pm 0.45\%$; TACRED Decision Fusion $42.83 \pm 0.30\%$, Hard Top-1 $42.91 \pm 0.51\%$.

### 1.3. Những thành phần chỉ mới nằm ở tài liệu hoặc chưa chạy hết
- **Chưa chạy thật:** `B3` ConPL full 8-task reproduction mới chỉ có smoke/audit (`runs/conpl/`), chưa có run `B4` CPL chính thức, chưa chạy hết `DS-10..DS-FULL` của Track B Scalability (`E013`/Track B chỉ mới có `ds05` trong `runs/tapta_scalability/`), và chưa chạy các ablation trong `workflow/EXPERIMENT_REGISTRY.md` (`tapa_no_tree`, `tapa_no_ema_kd` vốn còn ghi `Planned`).
- **Tồn đọng kỹ thuật đã phát hiện qua audit (`PROTOCOL_AUDIT.md`):**
  1. Có 2 hệ đánh giá khác nhau trong repo: `dataset-pipelines` (test split lớn 55,200 mẫu dùng ở `E001–E004`) và `ProtocolV1Dataset` (100 test/relation = 8,000 mẫu dùng ở `E007–E018`).
  2. Config TACRED cũ ghi `total_relations: 41` / `num_classes: 41` trong khi `data/protocol_v1/tacred/relation_name.txt` thực tế có 40 dòng.
  3. `CheckpointManager.save_checkpoint` trước đây ưu tiên `get_state_dict()` rồi mới đến `state_dict()` (đã được vá trong `cl_re_baselines/checkpoint.py`, nhưng `run_p4.py` cần kiểm tra đồng bộ).
  4. Một số tài liệu cũ ở `workflow/` (`STATUS.md`, `HANDOFF.md`, `EXPERIMENT_REGISTRY.md`) vẫn dừng ở ngày `2026-11-08` trước khi `P0–P5` được triển khai.

---

## 2. Lý Do Chuyển Hướng Sang DynaTree-CRE

Kết quả thực nghiệm `E012`–`E018` cho thấy ba giới hạn cấu trúc của việc tổ chức prompt theo **Task ID** (TAPTA):

1. **Task split ngẫu nhiên không phản ánh cấu trúc ngữ nghĩa của relations:** Kiểm tra trực tiếp `ProtocolV1Dataset(seed=2021)` cho thấy các quan hệ cùng miền bị xé lẻ qua nhiều task — ví dụ `sibling` và `child` nằm ở `Task 0`, `spouse` ở `Task 2`, `father` và `mother` ở `Task 3`; hay `mountain range` ở `Task 1`, `mouth of the watercourse` ở `Task 2`, `tributary`/`crosses` ở `Task 6`. Do đó, một "task prompt" $P^{\text{leaf}}_t$ phải phục vụ 10 quan hệ rời rạc về ngữ nghĩa, dẫn đến task centroid trong không gian BERT bị chồng lấn (`cos = 0.958`) và không thể chia sẻ kiến thức cross-task (`sibling` không chia sẻ được prompt với `father`/`mother`).
2. **Trần hiệu năng của Task-Routed P4 và P5:** Ngay cả khi `P4` giữ $L=8$ và đóng băng $P^{\text{root}}$, `P4` ($58.38\%$) vẫn không vượt qua baseline unprompted `D09` ($59.93\%$) hay `P2a` ($59.35\%$), và bản sửa lỗi `P5` (`E017`) lại giảm hiệu năng xuống $56.49\%$. Điều này cho thấy việc gắn 1 leaf prompt cho cả một task hỗn tạp là điểm nghẽn căn bản.
3. **Nhu cầu chuyển sang tổ chức tri thức theo ngữ nghĩa (Semantic Relation Hierarchy):** **DynaTree-CRE** thay thế phân cấp *Root → Task → Relation* bằng phân cấp động *Root → Semantic Cluster → Relation*, trong đó các cluster được hình thành tự động từ prototype của các relation đã quan sát, cho phép các relation gần nghĩa ở các task khác nhau cùng chia sẻ một cluster prompt và chuyển giao tri thức cross-task.

---

## 3. Phân Tích Tác Động & Migration Inventory

| File / Module | Current purpose | Status | Required action | Replacement | Reason |
|---|---|---|---|---|---|
| `src/data/protocol_v1_loader.py` | Loader FewRel/TACRED Protocol v1 Track A | `ACTIVE` | Tái sử dụng trực tiếp; xử lý sạch `total_relations` (40 cho TACRED) ở wrapper | — | Bảo toàn 100% dataset split & task order |
| `src/data/scalable_fewrel_loader.py` | Loader Track B nested shots & anti-leakage | `REUSABLE` | Giữ nguyên cho giai đoạn mở rộng Track B sau MVP | — | Không cần sửa cho MVP Track A |
| `dataset-pipelines/continual-relation-extraction/src/evaluation/*` | `PerformanceMatrix`, AA, AIA, Forgetting, BWT | `ACTIVE` | Tái sử dụng toàn bộ công thức chuẩn trong `src/dynatree_cre/metrics.py` | — | Giữ nguyên định nghĩa metric hiện tại |
| `experiments/continual-relation-baselines/*` | Runner, `CheckpointManager`, plotting của B0/E001–E004 | `REUSABLE` | Giữ nguyên để đối chứng B0/Upper/NCM | — | Bảo vệ khả năng tái lập baseline |
| `src/models/prompt_bert.py`, `task_router.py`, `p5_*.py` | Cài đặt P0–P5 của hướng TAPTA | `ARCHIVED` | Giữ nguyên 100%, không xóa hoặc sửa làm sai lệch kết quả cũ | `src/dynatree_cre/*` | Bảo toàn reproducibility của `E007–E018` |
| `experiments/run_p0_p1.py`..`run_p5.py`, `run_scalability_study.py` | Runner cho hướng TAPTA P0–P5 & scalability | `ARCHIVED` | Giữ nguyên; không sửa baseline logic | `experiments/run_dynatree_cre.py` | Tách biệt hoàn toàn runner hướng mới |
| `runs/*`, `results/*`, `reports/*`, `paper-reproductions/*` | Kết quả thực nghiệm cũ & mã nguồn baseline paper | `ARCHIVED` | Bảo vệ tuyệt đối, không ghi đè/xóa | `runs/dynatree_cre/*` | Tuân thủ Rule 2.1 bảo vệ nghiên cứu cũ |
| `docs/protocol-v1.md` | Đặc tả giao thức Protocol v1 | `ACTIVE` | Giữ nguyên làm chuẩn giao thức | — | Đảm bảo so sánh công bằng |
| `docs/specifications/architecture-freeze-memo.md` | Memo khóa kiến trúc TAPTA P4 | `SUPERSEDED` | Thêm YAML `status: superseded` & ghi chú lịch sử | `docs/research/RESEARCH_DIRECTION.md` | Chuyển trọng tâm sang DynaTree-CRE |
| `docs/research/p4-diagnosis/P5_SPECIFICATION.md` | Đặc tả TAPTA P5 | `SUPERSEDED` | Thêm YAML `status: superseded` & ghi chú lịch sử | `docs/research/methods/dynatree-cre/METHOD_SPEC.md` | Không dùng P5 làm hướng chính nữa |
| `docs/specifications/p4-hierarchical-prompt-tree-spec.md` | Đặc tả P4 TAPTA Tree | `ARCHIVED` | Thêm metadata `archived-comparator-spec` | `docs/research/methods/dynatree-cre/METHOD_SPEC.md` | Giữ làm comparator đối chứng |
| `docs/specifications/p0-p1-spec-protocol-v1.md`, `p2-router-spec.md`, `p3-soft-router-spec.md` | Đặc tả đối chứng P0–P3 | `REUSABLE` | Giữ nguyên làm comparator history | — | Có giá trị đối chứng với task-level baselines |
| `AGENTS.md`, `README.md`, `workflow/STATUS.md`, `workflow/DECISIONS.md` | Hướng dẫn repo & trạng thái | `UPDATE_REQUIRED` | Cập nhật hướng active sang DynaTree-CRE, giữ lịch sử TAPTA | — | Giúp agent phiên sau biết hướng active |
| `experiments/registry.yaml` | Sổ đăng ký `E001`–`E018` | `ACTIVE` | Giữ nguyên `E001–E018`; dùng prefix `DT-C*` / `DT-E*` cho DynaTree-CRE | — | Tránh xung đột ID thí nghiệm |
| `src/dynatree_cre/*` & `tests/dynatree_cre/*` | Package & test suite của DynaTree-CRE | `ACTIVE` (New) | Triển khai theo SDD + TDD cho M1–M6 | — | Kiến trúc mục tiêu mới |

---

## 4. Rủi Ro & Câu Hỏi Chưa Giải Quyết (Chuyển Giao Sang M1–M6)

1. **Anisotropy của Frozen BERT làm sụp đổ threshold `0.75`:** Cần xử lý bằng **T1-Centered Stationary Cosine Space** (RD-005) và kiểm chứng song song với raw cosine control.
2. **Độ dài Prompt $L=8$ vs $L=10$ và Classifier Bias:** Cần xuất đồng thời `ExpandableCosineClassifier` (freeze old weights) và `PromptedNCMClassifier`, giữ mặc định $L=8$ để so công bằng với `P0–P4`.
3. **Zero-Gradient Trap khi Top-K routing lúc train:** Nếu Top-K rơi hoàn toàn vào old frozen leaves tại $T_t \ge 2$, new leaves không nhận gradient. Cần đo `train_target_in_topk_rate` và `train_zero_prompt_grad_steps`, đồng thời cung cấp control `dense_soft` và biến thể `warmup_target_assist`.
4. **Hạn chế về Compute tại chỗ:** Máy local chạy Apple Silicon MPS; chỉ chạy test và smoke run (< 2 phút/run) tại chỗ, toàn bộ full 8-task / multi-seed runs cần chạy trên Google Colab GPU sau khi được phê duyệt Gate 5.
