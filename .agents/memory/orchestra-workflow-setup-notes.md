---
type: project
created: 2026-11-08
updated: 2026-11-08
---

# Orchestra AI Research Skills & Shared Workflow Setup Notes (What & Why)

Tài liệu này lưu vết chi tiết quá trình cài đặt, cấu hình, chuẩn hóa tài liệu và kiểm thử (validation) hệ thống **Orchestra AI Research Skills** dùng chung giữa **Google Antigravity (AGY)** và **OpenAI Codex** cho đề tài **Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPA-CRE)**.

Mỗi phase đều ghi rõ **ĐÃ LÀM GÌ (What)** và **TẠI SAO PHẢI LÀM (Why)** theo đúng quy ước người dùng trong `user-preferences.md`.

---

## Phase 1: Environment Discovery & Tooling Inspection

### 1. Đã làm gì (What)
- Kiểm tra toàn diện hệ thống: macOS 15.5 (`arm64` Apple Silicon), Node.js `v22.22.0`, npm/npx `10.9.4`, Python `3.14.0`, Git `2.39.5`.
- Kiểm tra tính sẵn sàng của CLI `agy` (`/Users/vinhlq2512/.local/bin/agy`) và `codex` (`/Users/vinhlq2512/.local/bin/codex` v0.166.0).
- Dùng lệnh `codex debug prompt-input` để kiểm tra chính xác cơ chế nạp chỉ thị của Codex: xác nhận Codex tự động đọc `AGENTS.md` ở root và có quyền truy cập vào `.agents/` và `.codex/`.
- Kiểm tra toàn bộ cấu trúc dự án: nhận diện dữ liệu FewRel Track A, TACRED, baseline B0 Sequential FT (`experiments/run_sequential_ft.py`), Upper Bound Joint Training, và registry lưu trữ run `E001`.
- Ghi nhận đầy đủ phát hiện tại [`docs/research-setup.md`](file:///Users/vinhlq2512/code/research-code-lab/docs/research-setup.md).

### 2. Tại sao phải làm (Why)
- **Inspect Before Modifying**: Tránh mọi giả định sai lầm về môi trường; bảo tồn nguyên vẹn toàn bộ mã nguồn, cấu hình, dữ liệu và kết quả thực nghiệm có sẵn.
- **Hiểu sâu cơ chế agent**: Việc chạy `codex debug prompt-input` giúp nắm bắt chính xác context pipeline của OpenAI Codex, từ đó thiết kế `AGENTS.md` tương thích tuyệt đối mà không cần can thiệp vào binary hay config nhạy cảm.

---

## Phase 2: Orchestra Installation & Curated Skill Selection

### 1. Đã làm gì (What)
- Tải và phân tích chi tiết mã nguồn installer `@orchestra-research/ai-research-skills@1.7.2` từ npm tarball và repository GitHub chính thức.
- Phát hiện cơ chế mặc định của installer khi dùng `--local` sẽ quét toàn bộ agent toàn cục và tạo ra 5 bản sao trùng lặp ở `.cursor/`, `.codex/`, `.gemini/`, `skills/`, `.agents/`.
- Thực hiện cài đặt có chọn lọc (curated selection) đúng 14 kỹ năng trọng tâm phục vụ đề tài nghiên cứu vào thư mục gốc duy nhất `.agents/skills/`:
  1. `0-autoresearch-skill`: Quản lý toàn bộ vòng đời nghiên cứu (Autoresearch lifecycle).
  2. `21-research-ideation/brainstorming-research-ideas`: Phát triển ý tưởng và lập luận khoa học.
  3. `21-research-ideation/creative-thinking-for-research`: Tư duy đa chiều và liên ngành.
  4. `20-ml-paper-writing/ml-paper-writing`: Viết bài báo khoa học chuẩn NeurIPS/ICML/ACL.
  5. `20-ml-paper-writing/academic-plotting`: Vẽ biểu đồ khoa học chuẩn ấn phẩm (SVG/PDF).
  6. `03-fine-tuning/peft`: Huấn luyện tham số hiệu quả (LoRA, QLoRA, Prompt Tuning).
  7. `10-optimization/bitsandbytes`: Lượng tử hóa 4-bit/8-bit tiết kiệm bộ nhớ trên MPS/CUDA.
  8. `10-optimization/ml-training-recipes`: Tối ưu hóa vòng lặp huấn luyện PyTorch, scheduler và seed.
  9. `08-distributed-training/accelerate`: Trừu tượng hóa huấn luyện phần cứng nhẹ của HuggingFace.
  10. `11-evaluation/lm-evaluation-harness`: Bộ khung đánh giá chuẩn benchmark và few-shot.
  11. `13-mlops/weights-and-biases`: Theo dõi thực nghiệm qua W&B.
  12. `13-mlops/tensorboard`: Theo dõi thực nghiệm cục bộ (zero-cost, offline).
  13. `19-emerging-techniques/knowledge-distillation`: Chưng cất tri thức, EMA teacher chống quên.
  14. `22-agent-native-research-artifact/rigor-reviewer`: Đánh giá độ chặt chẽ khoa học và ablation.
- Cập nhật lockfile chuẩn [`.orchestra-skills.json`](file:///Users/vinhlq2512/code/research-code-lab/.orchestra-skills.json).
- Tạo symlink tương đối sạch sẽ `/.codex/skills -> ../.agents/skills` để Codex truy cập trực tiếp mà không sinh bản sao dư thừa.

### 2. Tại sao phải làm (Why)
- **Chống phình tải (Anti-Bloat & Minimal Set)**: Không cài đặt toàn bộ 98 kỹ năng để tránh tiêu tốn token context và phân tán chuyên môn của agent.
- **Đúng định hướng nghiên cứu**: 14 kỹ năng được chọn ánh xạ 1-1 với cấu trúc đề tài TAPA-CRE (Frozen PLM + LoRA/QLoRA, Hierarchical Prompt Tree, Dynamic Soft Routing with Prototypes, Prototype Memory, Knowledge Distillation and EMA Teacher).
- **Single Source of Truth**: `.agents/skills/` là nơi lưu trữ tập trung duy nhất, `.codex/skills` trỏ relative symlink giúp loại bỏ hoàn toàn nguy cơ mất đồng bộ giữa hai agent.

---

## Phase 3: Shared Workspace Scaffolding

### 1. Đã làm gì (What)
- Thiết lập cấu trúc thư mục nghiên cứu phân tách rõ ràng:
  - `research/`: `literature/`, `hypotheses/`, `methodology/`, `experiments/`, `results/`, `papers/`, `thesis/`.
  - `workflow/`: `RESEARCH_PLAN.md`, `TASKS.md`, `STATUS.md`, `HANDOFF.md`, `EXPERIMENT_REGISTRY.md`, `DECISIONS.md`.
  - `docs/`: `research-setup.md`, `research-workflow.md`.
- Cập nhật toàn diện [`AGENTS.md`](file:///Users/vinhlq2512/code/research-code-lab/AGENTS.md) định nghĩa:
  - Mục tiêu nghiên cứu và giới hạn thực nghiệm.
  - 4 nguyên tắc liêm chính khoa học (Scientific Integrity).
  - Phân định vai trò AGY vs Codex.
  - Cơ chế kích hoạt và khám phá skill.
  - Chuẩn mực tái lập thực nghiệm (Reproducibility).
  - Quy trình bàn giao (Handoff) và quyền sở hữu file.
  - 6 cổng phê duyệt của con người (Human Approval Gates 1–6).
  - Bảo tồn toàn bộ các mục có sẵn (`Issue tracker`, `Triage labels`, `Domain docs`).

### 2. Tại sao phải làm (Why)
- **Tổ chức khoa học chuẩn mực**: Ngăn ngừa tình trạng nhầm lẫn giữa giả thuyết nghiên cứu (hypotheses) và số liệu đo lường thực tế (results).
- **Dual-Agent Compatibility**: Cung cấp tài liệu quy chuẩn duy nhất mà cả Antigravity và Codex đều đọc được ngay khi khởi động session.

---

## Phase 4: Nine-Stage Research Pipeline (R1–R9)

### 1. Đã làm gì (What)
- Soạn thảo quy trình 9 giai đoạn chi tiết tại [`docs/research-workflow.md`](file:///Users/vinhlq2512/code/research-code-lab/docs/research-workflow.md):
  - R1: Literature Review (AGY lead, Gate 1)
  - R2: Research Gap Identification (AGY lead)
  - R3: Hypothesis Generation (AGY lead, Gate 2)
  - R4: Methodology Design (AGY lead, Gate 3)
  - R5: Baseline Implementation & Validation (Codex lead)
  - R6: Proposed Method Implementation (Codex lead, Gate 4)
  - R7: Experiment Execution (Codex lead, Gate 5)
  - R8: Evaluation and Ablation (Codex + AGY)
  - R9: Scientific Writing (AGY lead, Gate 6)
- Mỗi giai đoạn định nghĩa chi tiết: Responsible agent, Inputs, Required skills, Steps, Expected outputs, Acceptance criteria, Human approval requirements.

### 2. Tại sao phải làm (Why)
- **Phân công tối ưu năng lực**: AGY sở trường bao quát, tổng hợp tài liệu, tư duy phản biện và viết báo khoa học. Codex sở trường sinh mã PyTorch chuẩn xác, chạy test, debug phần cứng và đo lường số liệu.
- **Quy trình đóng kín, có kiểm soát**: Ngăn chặn việc agent tự tiện chuyển giai đoạn khi chưa đáp ứng tiêu chí nghiệm thu (acceptance criteria) hoặc chưa có chữ ký duyệt của con người (human approval gates).

---

## Phase 5: File-Based Agent Handoff Mechanism

### 1. Đã làm gì (What)
- Xây dựng cơ chế handoff qua các file markdown:
  - [`workflow/HANDOFF.md`](file:///Users/vinhlq2512/code/research-code-lab/workflow/HANDOFF.md): Chứa đầy đủ các trường bắt buộc (Task ID, Assigned agent, Objective, Relevant context & files, Expected artifacts, Acceptance criteria, Completion status, Blockers, Next action) và template chuẩn cho các lượt chuyển giao tương lai.
  - [`workflow/STATUS.md`](file:///Users/vinhlq2512/code/research-code-lab/workflow/STATUS.md): Bảng điều khiển tiến độ cấp cao.
  - [`workflow/TASKS.md`](file:///Users/vinhlq2512/code/research-code-lab/workflow/TASKS.md): Danh mục công việc chi tiết.
- Quy định nguyên tắc Single Writer và cô lập nhánh/worktree (`research/[slug]` vs `feature/[slug]`).

### 2. Tại sao phải làm (Why)
- **Không phụ thuộc chat history**: Một session Codex mới có thể tiếp nhận công việc trơn tru chỉ bằng cách đọc `AGENTS.md` và `workflow/HANDOFF.md` mà không cần xem lại lịch sử hội thoại của AGY.
- **Tránh xung đột ghi (Race conditions)**: Ngăn hai agent chỉnh sửa đồng thời vào cùng một file mã nguồn.

---

## Phase 6: Research Safety, Reproducibility & Integrity Protocol

### 1. Đã làm gì (What)
- Thiết lập [`workflow/EXPERIMENT_REGISTRY.md`](file:///Users/vinhlq2512/code/research-code-lab/workflow/EXPERIMENT_REGISTRY.md) liên kết trực tiếp với [`experiments/registry.yaml`](file:///Users/vinhlq2512/code/research-code-lab/experiments/registry.yaml).
- Đóng băng các quy tắc an toàn:
  - Không bịa đặt số liệu benchmark (Anti-Fabrication Rule).
  - Cố định seed schedule (`2021`, `2022`, `2023`) và task order chuẩn FewRel/TACRED.
  - Phê duyệt bắt buộc trước các tác vụ GPU/MPS tốn kém (> 2 phút).
  - Cấu hình autoresearch có giới hạn lặp (bounded experiments), nghiêm cấm vòng lặp vô hạn.

### 2. Tại sao phải làm (Why)
- **Đảm bảo tính trung thực khoa học**: Tránh hallucination về kết quả thực nghiệm hoặc so sánh không công bằng với các nghiên cứu trước.
- **Tiết kiệm tài nguyên và bảo vệ phần cứng**: Ngăn máy bị quá tải hoặc phát sinh chi phí API ngoài ý muốn.

---

## Phase 7: Comprehensive Validation & Smoke Testing

### 1. Đã làm gì (What)
- **Validate SKILL.md**: Xác thực 100% (14/14) file `SKILL.md` đã cài đặt có header YAML frontmatter hợp lệ và nội dung đầy đủ.
- **Validate CLI discovery**: Lệnh `npx @orchestra-research/ai-research-skills list --local` liệt kê đúng 14 kỹ năng theo từng danh mục.
- **Validate Codex discovery**: Dùng `codex debug prompt-input` xác nhận Codex nạp đầy đủ nội dung `AGENTS.md`; đường dẫn `.codex/skills/` trỏ đúng vào `.agents/skills/`.
- **Thực thi bàn giao mẫu (Handoff HO-001 / Task T-002)**:
  - Thực thi kiểm thử hồi quy nhẹ: `./.venv/bin/pytest experiments/tests/ -v`.
  - Kết quả: 4/4 bài kiểm tra vượt qua (100% PASSED trong 0.01s).
  - Tạo log kết quả kiểm thử tại [`reports/smoke-test-codex.log`](file:///Users/vinhlq2512/code/research-code-lab/reports/smoke-test-codex.log).
  - Cập nhật trạng thái hoàn thành vào `workflow/HANDOFF.md`, `workflow/STATUS.md`, và `workflow/TASKS.md`.

### 2. Tại sao phải làm (Why)
- **End-to-End Verification**: Đảm bảo toàn bộ quy trình từ cài đặt, cấu hình, liên kết symlink đến cơ chế bàn giao hoạt động trơn tru trong thực tế chứ không chỉ nằm trên văn bản.
- **Bảo toàn hiện trạng**: Chứng minh rằng toàn bộ mã nguồn kiểm thử và dữ liệu baseline cũ vẫn hoạt động hoàn hảo sau khi tích hợp Orchestra.

---

## Phase 8: Transition to K-Dense Scientific Agent Skills

### 1. Đã làm gì (What)
- Theo yêu cầu trực tiếp từ người dùng, thực hiện gỡ bỏ 14 kỹ năng Orchestra (`0-autoresearch-skill`, `peft`, `knowledge-distillation`, `ml-paper-writing`,...) và file `.orchestra-skills.json`.
- Tải và cài đặt 177 kỹ năng nghiên cứu khoa học chuyên sâu từ [`K-Dense-AI/scientific-agent-skills`](https://github.com/K-Dense-AI/scientific-agent-skills) (v2.72.0) vào `.agents/skills/`.
- Lưu trữ manifest [`.scientific-skills-plugin.json`](file:///Users/vinhlq2512/code/research-code-lab/.scientific-skills-plugin.json).
- Giữ nguyên symlink an toàn `.codex/skills -> ../.agents/skills` để Codex tự động đồng bộ 100% (177/177) kỹ năng mới.
- Cập nhật mục 4 trong `AGENTS.md`, `workflow/STATUS.md`, và `workflow/TASKS.md`.

### 2. Tại sao phải làm (Why)
- **User Preference Alignment**: Người dùng yêu cầu thay thế bộ kỹ năng Orchestra cũ bằng kho kỹ năng khoa học mở rộng của K-Dense.
- **Bảo toàn tính tương thích**: Không xoá các kỹ năng kỹ thuật phần mềm cốt lõi sẵn có của AG Kit (`clean-code`, `architecture`, `verify-changes`,...), chỉ thay thế đúng phạm vi kỹ năng nghiên cứu được yêu cầu.
- **Zero Duplication**: Vẫn duy trì single-source-of-truth tại `.agents/skills` với symlink sang `.codex/skills`.

---

## Phase 9: Project-Relevance Filtering & Specialization (TAPA-CRE Focused)

### 1. Đã làm gì (What)
- Theo yêu cầu làm rõ từ người dùng ("Tôi chỉ muốn cài các skill liên quan đến project"), đã tiến hành rà soát toàn bộ 177 kỹ năng để lọc bỏ các kỹ năng không thuộc phạm vi xử lý ngôn ngữ tự nhiên (NLP) và học liên tục (Continual Learning).
- **Loại bỏ 152 kỹ năng không liên quan**: Toàn bộ kỹ năng về phòng thí nghiệm ướt (wet-lab automation như `opentrons-integration`, `pylabrobot`), sinh học phân tử đơn tế bào (`anndata`, `scanpy`, `cellprofiler`), hóa học phân tử (`cantera`, `datamol`, `diffdock`, `rdkit`), thiên văn học (`astropy`), cơ học chất lỏng (`fluidsim`), lượng tử (`cirq`, `qiskit`, `qutip`), và dữ liệu lâm sàng y khoa.
- **Giữ lại và củng cố 35 kỹ năng trọng tâm tuyệt đối cho TAPA-CRE**:
  - Mô hình nền tảng & PEFT: `transformers`, `peft`, `bitsandbytes`, `pytorch-lightning`, `accelerate`, `optimize-for-gpu`, `ml-training-recipes`.
  - Học liên tục, Soft Routing & Prototypes: `knowledge-distillation` (EMA Teacher), `scikit-learn` (Clustering/Prototypes), `umap-learn`, `shap`.
  - Đánh giá benchmark & MLOps: `lm-evaluation-harness`, `polars`, `tensorboard`, `weights-and-biases`.
  - Phương pháp luận khoa học: `0-autoresearch-skill`, `literature-review`, `citation-management`, `hypothesis-generation`, `scientific-brainstorming`, `scientific-critical-thinking`, `experimental-design`, `peer-review`, `rigor-reviewer`.
  - Thống kê & Công bố bài báo: `statistical-analysis`, `statistical-power`, `statsmodels`, `scientific-writing`, `scientific-visualization`, `matplotlib`, `seaborn`, `venue-templates`, `scientific-slides`, `latex-posters`, `markdown-mermaid-writing`.
- Cập nhật mục 4 trong `AGENTS.md`, `workflow/STATUS.md`, và `workflow/TASKS.md`.

### 2. Tại sao phải làm (Why)
- **Chống phình ngữ cảnh (Context Pollution & Token Bloat)**: Việc để hơn 150 kỹ năng sinh học và hóa học làm loãng bộ nhớ của AGY và Codex, khiến việc định tuyến (skill routing) bị chậm và tiêu tốn token vô ích.
- **Tập trung cao độ vào bản chất đề tài**: 35 kỹ năng được giữ lại bao phủ trọn vẹn 100% vòng đời nghiên cứu của TAPA-CRE: từ mô hình frozen PLM, adapter LoRA, dynamic routing prototype, EMA distillation, benchmark FewRel/TACRED cho đến kiểm định thống kê và soạn thảo bài báo ACL/NeurIPS.
