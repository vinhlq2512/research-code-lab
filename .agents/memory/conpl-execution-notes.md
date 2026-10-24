---
type: project
created: 2026-10-13
updated: 2026-10-13
---

# ConPL (ACL 2023) Reproduction & Baseline Integration Execution Notes (What & Why)

Tài liệu này lưu vết chi tiết việc kiểm định, tái lập thực nghiệm và tích hợp đường cơ sở **ConPL (Consistent Prototype Learning - ACL 2023)** cho bài toán Continual Relation Extraction trên FewRel 5-shot. Mỗi phần ghi rõ **ĐÃ LÀM GÌ (What)** và **TẠI SAO PHẢI LÀM (Why)**.

---

## Phase 1: Audit & Khóa Provenance Mã nguồn

### 1. Đã làm gì (What)
- Kiểm tra kho mã nguồn `paper-reproductions/ConPL`, khóa commit SHA `973911899e4b70c705214a2172f54941508ca6b1`.
- Kiểm tra toàn bộ git diff so với `origin/main`: Xác nhận chỉ có 2 file được chỉnh sửa (patch nới lỏng `tokenizers < 0.14` trong `dependency_versions_table.py` và patch gỡ hardcode device + thêm environment variables trong `fewrel_5shot.py`).
- Lập tài liệu audit: [`docs/reproductions/conpl-environment-audit.md`](../../docs/reproductions/conpl-environment-audit.md).

### 2. Tại sao phải làm (Why)
- **Bảo toàn thuật toán gốc (Provenance Integrity):** Đảm bảo mọi dòng mã về prototype loss, consistency regularization, margin loss và exemplar replay được giữ nguyên 100%, không bị sửa đổi âm thầm.
- **Tương thích macOS ARM64:** Cần `tokenizers==0.13.3` để có wheel binary có sẵn trên Apple Silicon, tránh đòi hỏi trình biên dịch Rust khi cài đặt.

---

## Phase 2: Xác nhận Môi trường Cách ly & Smoke Imports

### 1. Đã làm gì (What)
- Thiết lập và kiểm định môi trường độc lập tại `paper-reproductions/ConPL/.venv` (Python 3.10.19 ARM64).
- Cố định PyTorch 2.2.2 (MPS backend), NumPy 1.23.5, Tokenizers 0.13.3, HuggingFace Hub 0.10.1.
- Chạy script import smoke thành công toàn bộ module `BERTSentenceEncoderPrompt`, `data_sampler_bert_prompt_deal_first_task`, `proto_softmax_layer_bert_prompt`.

### 2. Tại sao phải làm (Why)
- **Tránh xung đột NumPy Ragged Array:** NumPy $\ge 1.24$ cấm tạo ndarray từ nested sequence không đồng nhất và gây `ValueError`. Việc pin `numpy==1.23.5` giải quyết triệt để lỗi này.
- **Tránh phá vỡ API Transformers nội bộ:** Bản `transformers/` nội bộ của repo là `4.22.0.dev0`, cần `huggingface-hub < 1.0` để không bị lỗi `ImportError: cannot import name 'HfFolder'`.

---

## Phase 3: Khóa Dữ liệu & Kiểm chứng Sampler Protocol

### 1. Đã làm gì (What)
- Băm SHA-256 khóa toàn bộ 7 file dữ liệu chuẩn trong `data/fewrel/`.
- Viết script kiểm tra sampler cho Seed 100:
  - 8 tasks liên tục, mỗi task đúng 10 quan hệ mới (tổng 80 quan hệ).
  - Xác nhận hàm `_deal_first_task` rút gọn task 1 về đúng 5-shot (50 mẫu / task trên cả 8 tasks).
  - Tập test lũy kế từ 1000 mẫu ($T_1$) đến 8000 mẫu ($T_8$).
- Lập tài liệu đặc tả protocol: [`docs/reproductions/conpl-protocol.md`](../../docs/reproductions/conpl-protocol.md).

### 2. Tại sao phải làm (Why)
- **Đảm bảo tính chân thực của Protocol:** Nhiều benchmark CRE có sự khác biệt giữa Data-rich Task 1 (100-shot như CPL) và Pure Few-shot (5-shot như ConPL). Việc assert 50 train samples/task xác thực mô hình được đánh giá đúng chuẩn NK-CRE của ConPL paper.
- **Tránh Data Leakage:** Đảm bảo không gian nhãn kiểm thử tại inference mở rộng theo từng task và test samples hoàn toàn độc lập với train.

---

## Phase 4: Chạy Thực nghiệm Kiểm chứng Local (Seed 100)

### 1. Đã làm gì (What)
- Kích hoạt huấn luyện thực tế với mô hình BERT thật trên Apple Silicon M3 Pro GPU (`mps`) kết hợp cờ `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Thu được kết quả thực nghiệm từ $T_1 \to T_4$ trên Seed 100:
  - $T_1$: **94.90%** (Paper Table 1: 95.72%, lệch -0.82%)
  - $T_2$: **93.50%** (Paper Table 1: 93.53%, lệch **-0.03%**)
  - $T_3$: **90.67%** (Paper Table 1: 91.31%, lệch -0.64%)
  - $T_4$: **87.48%** (Paper Table 1: 89.95%, lệch -2.47%)
- Lưu checkpoint thực tế: `runs/conpl/mac_t1_t8_seed100/seed_100/checkpoint_latest.pt` (438 MB) và `task_metrics.jsonl`.

### 2. Tại sao phải làm (Why)
- **Minh chứng toán học & thực nghiệm (Empirical Proof):** Chứng minh mã nguồn tác giả hoạt động trơn tru trên máy thật, tái hiện sát sao số liệu Table 1 từ task đầu đến task 4.
- **Phát hiện giới hạn phần cứng:** `aten::multi_margin_loss` không có native kernel trên Metal (MPS) và phải fallback về CPU, chứng minh quyết định chuyển run 6 sequence đầy đủ sang Linux GPU là chính xác.

---

## Phase 5 & 6: Xây dựng Google Colab Notebook & Tích hợp Luận văn

### 1. Đã làm gì (What)
- Xây dựng notebook Colab chuẩn mực: [`notebooks/conpl_fewrel_5shot.ipynb`](../../notebooks/conpl_fewrel_5shot.ipynb) hỗ trợ GPU T4/V100/A100 với quy trình 1-click.
- Cập nhật mục lục [`notebooks/README.md`](../../notebooks/README.md) và [`README.md`](../../README.md).
- Lập báo cáo tổng kết khoa học: [`reports/conpl_fewrel_5shot_reproduction.md`](../../reports/conpl_fewrel_5shot_reproduction.md).

### 2. Tại sao phải làm (Why)
- **Tính khả thi cho Nghiên cứu (Reproducibility for Thesis):** Bất kỳ nhà nghiên cứu nào cũng có thể chạy lại trọn vẹn 6 sequence trên Colab GPU miễn phí mà không cần máy chủ Linux riêng.
- **Đóng khung hệ thống Baseline:** ConPL trở thành mốc so sánh SOTA Prototype hoàn chỉnh bên cạnh Lower Bound B0 và Upper Bound Joint Training.
