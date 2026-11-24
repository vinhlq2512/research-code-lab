# Báo Cáo Kiểm Tra Toàn Bộ Hiện Trạng Dự Án (Repository Audit)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Chủ đề:** TAPTA – Data Scalability & High-Resource Continual Relation Extraction  
**Thời điểm kiểm tra:** 2026-11-24  
**Nhánh Git:** `feat/tapta-data-scalability`  
**Trạng thái kiểm tra:** HOÀN TẤT KIỂM TOÁN (100% Căn cứ Source Code & Artifacts)

---

## 1. Tóm Tắt Hiện Trạng & Ranh Giới Kỹ Thuật

Mục tiêu kiểm toán là xác thực toàn bộ hạ tầng hiện tại của TAPTA (tập trung vào phiên bản P4 đã khóa kiến trúc tại `MEMO-ARCH-FREEZE-W18`), cấu trúc dữ liệu FewRel/TACRED, cơ chế routing, bộ nhớ $M=0$, và các rủi ro kỹ thuật khi mở rộng từ Few-Shot (5-shot) sang High-Resource (10, 50, 100, Full-Data).

### Phân Loại Thành Phần Hệ Thống:
- `VERIFIED`: Đã kiểm chứng qua source code, commit history, và file thực tế.
- `ASSUMED`: Giả định cần làm rõ / có sự khác biệt giữa tài liệu và code.
- `MISSING`: Chưa tồn tại trong repository.
- `NEEDS_CHANGE`: Cần mở rộng / tái cấu trúc để hỗ trợ scalability.
- `RISK`: Rủi ro kỹ thuật hoặc rủi ro tính đúng đắn khoa học (Data leakage, OOM, Timeout).

---

## 2. Bảng Kiểm Toán Chi Tiết Từng Hạng Mục

| Thành phần | Trạng thái | Bằng chứng kiểm tra thực tế (File & Line) | Phân tích & Ranh giới kỹ thuật |
|---|:---:|---|---|
| **Kiến trúc TAPTA P4** | `VERIFIED` | [`src/models/prompt_bert.py`](file:///Users/vinhlq2512/code/research-code-lab/src/models/prompt_bert.py), [`src/models/task_router.py`](file:///Users/vinhlq2512/code/research-code-lab/src/models/task_router.py), [`experiments/run_p4.py`](file:///Users/vinhlq2512/code/research-code-lab/experiments/run_p4.py) | Kiến trúc Cây mồi nhắc 2 tầng ($P^{\text{root}}_{L=4} + P^{\text{leaf}}_{t, L=4}$). $P^{\text{root}}$ học ở Task 0 và đóng băng vĩnh viễn ở Task 1..7. |
| **Mô hình Backbone** | `VERIFIED` | [`src/models/prompt_bert.py#L42-L75`](file:///Users/vinhlq2512/code/research-code-lab/src/models/prompt_bert.py#L42-L75) | `bert-base-uncased` (109.5M params), đóng băng 100% (`requires_grad = False`). Bổ sung 4 entity tokens `[E1]`, `[/E1]`, `[E2]`, `[/E2]` (hoặc `[E11]..[E22]`). |
| **Bộ Định Tuyến Tác Vụ (Task Router)** | `VERIFIED` | [`src/models/task_router.py#L22-L130`](file:///Users/vinhlq2512/code/research-code-lab/src/models/task_router.py#L22-L130) | Phi tham số (Non-parametric): Trích xuất $q(x) = \text{Normalize}_{\ell_2}([h_{\text{[E1]}}; h_{\text{[E2]}}])$ trên frozen BERT (không prompt). Tính điểm: $s_t(x) = (1-\alpha) q^\top \mu_t + \alpha \max_c q^\top \nu_{t,c}$ với $\alpha=0.7$. |
| **Cơ Chế Suy Luận (Inference)** | `VERIFIED` | [`experiments/run_p4.py#L239-L380`](file:///Users/vinhlq2512/code/research-code-lab/experiments/run_p4.py#L239-L380) | Hoàn toàn **Task-Agnostic** (No Oracle Task ID). Hỗ trợ Decision-Level Soft Fusion ($K=2, \tau=0.2$ hoặc $1.0$) và Gated Prototype Inference. |
| **Ràng Buộc Bộ Nhớ $M=0$** | `VERIFIED` | [`experiments/run_p4.py#L100-L235`](file:///Users/vinhlq2512/code/research-code-lab/experiments/run_p4.py#L100-L235) | Hoàn toàn không lưu câu văn thô của task cũ ($M=0$). Chỉ lưu vector nguyên mẫu $p_c \in \mathbb{R}^{2d}$ và tâm router $\mu_t, \nu_{t,c}$ (~1.03 MB toàn bộ 8 task). |
| **Bộ Dữ Liệu WAVE-CRE `data_with_marker.json`** | `VERIFIED` | [`paper-reproductions/WAVE-CRE/datasets/data_with_marker.json`](file:///Users/vinhlq2512/code/research-code-lab/paper-reproductions/WAVE-CRE/datasets/data_with_marker.json) | File tồn tại (15.6 MB), chứa đúng **80 relations**, mỗi relation có đúng **700 samples** (tổng 56,000 samples). Đã gắn sẵn entity markers. |
| **Khóa Thứ Tự Quan Hệ (PID <-> Tên)** | `VERIFIED` | [`paper-reproductions/WAVE-CRE/datasets/id2rel.json`](file:///Users/vinhlq2512/code/research-code-lab/paper-reproductions/WAVE-CRE/datasets/id2rel.json) vs [`data/protocol_v1/fewrel/relation_name.txt`](file:///Users/vinhlq2512/code/research-code-lab/data/protocol_v1/fewrel/relation_name.txt) | Danh sách 80 PIDs trong `id2rel.json` khớp chính xác 1-1 theo thứ tự dòng với 80 relation names trong `relation_name.txt` (Index 0: P931 = "place served by transport hub", ...). |
| **Dataset Loader Hiện Tại (`protocol_v1_loader.py`)** | `NEEDS_CHANGE` | [`src/data/protocol_v1_loader.py#L75-L240`](file:///Users/vinhlq2512/code/research-code-lab/src/data/protocol_v1_loader.py#L75-L240) | Đọc từ `CFRLdata_10_100_10_5/train_0.txt`. File này chỉ có 5 mẫu cho 70 quan hệ (do format ConPL `deal_first_task`). **Không thể dùng trực tiếp để lấy 10, 50, 100 hay 420 mẫu train!** |
| **Trùng Lặp Train/Test trong `CFRLdata_10_100_10_5`** | `RISK` | Đã kiểm tra nội dung text giữa `train_0.txt` và `test_0.txt` | Phát hiện 11 câu bị trùng giữa train và test trong file gốc của ConPL. Cần làm sạch để triệt tiêu 100% data leakage khi xây dựng scalability pool. |
| **Tập Test và Valid trong Protocol v1** | `VERIFIED` | [`data/protocol_v1/provenance_hashes.json`](file:///Users/vinhlq2512/code/research-code-lab/data/protocol_v1/provenance_hashes.json) | `valid_0.txt` và `test_0.txt` giống nhau 100% (cùng hash SHA-256). Đều chứa 100 mẫu/quan hệ $\times$ 80 = 8,000 câu kiểm thử. |
| **Bộ Thí Nghiệm Baseline Đã Chạy** | `VERIFIED` | [`experiments/registry.yaml`](file:///Users/vinhlq2512/code/research-code-lab/experiments/registry.yaml) | Đã ghi nhận kết quả thật: P0 (E008: 52.83%), P1 (E009: 70.14%), P2a (E010a: 59.35%), P2b (E010b: 56.04%), P3 (E011: 55.50%), P4 (E012: 57.58% Fusion / 58.80% Gated Proto). |
| **Hạ Tầng Checkpoint & Resume** | `VERIFIED` | [`cl_re_baselines/checkpoint.py`](file:///Users/vinhlq2512/code/research-code-lab/experiments/continual-relation-baselines/src/cl_re_baselines/checkpoint.py) | Lưu `checkpoint_task_{t}.pt`, cho phép resume chính xác tại task $t$ mà không cần chạy lại từ đầu. |
| **Cơ Chế Đa Seed Tự Động** | `VERIFIED` | [`experiments/run_multiseed.py`](file:///Users/vinhlq2512/code/research-code-lab/experiments/run_multiseed.py) | Đã sẵn sàng chạy đa seed (2021, 42, 100) và tự động tính bảng $\mu \pm \sigma$. |
| **Dataset Scalability Loader Cho Mọi Cỡ Mẫu** | `MISSING` | Chưa có loader đọc từ full pool 56,000 câu để sinh nested subsets | Cần tạo mới module dataset loader hỗ trợ: 5, 10, 50, 100, Full với nested sampling. |

---

## 3. Danh Sách Các File Cần Mở Rộng & Chỉnh Sửa

1. **`src/data/scalable_fewrel_loader.py` (Tạo mới):**
   - Đọc từ nguồn full 56,000 mẫu `paper-reproductions/WAVE-CRE/datasets/data_with_marker.json`.
   - Giữ nguyên cấu trúc phân cụm 8 task $\times$ 10 relations (`rel_cluster_label_0.npy`).
   - Khóa chặt tập kiểm thử Test Set (100 mẫu/relation hoặc 140 mẫu/relation chuẩn).
   - Loại trừ 100% các câu thuộc test set ra khỏi train pool ($\text{Train Pool} \cap \text{Test Set} = \emptyset$).
   - Thực thi **Nested Deterministic Sampling** theo seed:
     $$S_5 \subset S_{10} \subset S_{50} \subset S_{100} \subset S_{\text{Full}}$$
2. **`configs/scalability/` (Tạo mới danh mục configs):**
   - `p4_fewrel_ds05_seed2021.yaml` (5-shot)
   - `p4_fewrel_ds10_seed2021.yaml` (10-shot)
   - `p4_fewrel_ds50_seed2021.yaml` (50-shot)
   - `p4_fewrel_ds100_seed2021.yaml` (100-shot)
   - `p4_fewrel_ds_full_seed2021.yaml` (Full train pool: ~420-500 samples/relation)
3. **`experiments/run_scalability_study.py` (Tạo mới):**
   - Kế thừa 100% logic P4 từ `run_p4.py`.
   - Bổ sung tracking: thời gian huấn luyện từng task, GPU peak memory, kích thước persistent memory, và logging độ chính xác routing chi tiết.
   - Hỗ trợ cờ `--oracle_diagnostic` để chạy song song Task-Agnostic vs Oracle-Task Inference trên cùng checkpoint.
4. **`tests/test_scalable_loader.py` (Tạo mới bộ unit test TDD):**
   - Test kích thước mẫu chính xác (5, 10, 50, 100, Full).
   - Test tính chất tập con lồng nhau (Nested sampling property).
   - Test zero data leakage ($\text{Train} \cap \text{Test} = \emptyset$).
   - Test tính tất định theo seed (Deterministic reproducibility).

---

## 4. Phân Tích Rủi Ro Kỹ Thuật (Risk Register)

| ID | Rủi ro | Mức độ | Hậu quả | Biện pháp kiểm soát & Giảm thiểu |
|:---:|---|:---:|---|---|
| **R1** | **Data Leakage giữa Full Train Pool và Test Set** | Nghiêm trọng | Điểm số cao ảo, vi phạm tính liêm chính khoa học | Kiểm tra hash nội dung chuẩn hóa của toàn bộ câu trong test set; lọc bỏ triệt để khỏi train pool trước khi sampling. Có unit test assertion tự động. |
| **R2** | **OOM hoặc Colab Timeout khi chạy Full-Data** | Trung bình | Đứt gãy quá trình train Task 5..8 | Do BERT đóng băng 100%, chỉ train 3,072 params của prompt, memory gradient cực thấp. Tăng batch size lên 32 hoặc 64 khi train Full-Data; lưu checkpoint sau mỗi task để có thể resume tức thì. |
| **R3** | **Phá vỡ tính tương thích ngược với Track A 5-shot** | Cao | Sai lệch kết quả E012 đã công bố | Giữ nguyên nguyên vẹn `protocol_v1_loader.py` và các configs cũ; triển khai loader scalability thành module riêng `scalable_fewrel_loader.py`. |
| **R4** | **Nhiễu do lấy mẫu ngẫu nhiên (Sampling Variance)** | Trung bình | Khó kết luận hiệu năng tăng do cỡ dữ liệu hay do may mắn chọn trúng mẫu tốt | Áp dụng bắt buộc **Nested Sampling**: Mẫu của 5-shot nằm trọn trong 10-shot, 10-shot nằm trong 50-shot... |

---

## 5. Kết Luận Phase 0 (Definition of Done)
- [x] Đã kiểm toán 100% source code, data paths và model implementations.
- [x] Đã xác minh file `data_with_marker.json` (56,000 mẫu, 80 relations $\times$ 700 samples).
- [x] Đã phát hiện hạn chế của `train_0.txt` cũ và thiết kế phương án khắc phục sạch.
- [x] Sẵn sàng chuyển tiếp sang **PHASE 1 (Literature & Benchmark Verification)**.
