# Audit Môi trường & Mã nguồn ConPL (ACL 2023)

- **Ngày audit:** 2026-10-13  
- **Mục tiêu:** Khóa provenance, ghi nhận cấu trúc thư viện, dependency, và các biến cấu hình phục vụ tái lập ConPL trên FewRel 5-shot.

---

## 1. Thông tin Kho mã nguồn (Git Provenance)

- **Repository:** `https://github.com/XiudiChen/ConPL`
- **Vị trí checkout trong repo:** `paper-reproductions/ConPL`
- **Upstream Commit SHA:** `973911899e4b70c705214a2172f54941508ca6b1` (Update README.md)
- **Branch làm việc:** `codex/conpl-mac-colab-reproduction`

### Rà soát thay đổi (Git Diff vs `origin/main`)

Chỉ có 2 tệp được chỉnh sửa có chủ đích để hỗ trợ điều khiển runtime và tương thích hệ thống:

1. **`transformers/dependency_versions_table.py`:**
   - Nới lỏng chặn phiên bản `tokenizers`: `"tokenizers": "tokenizers>=0.11.1,!=0.11.3,<0.14"` (gốc là `<0.13`).
   - Rationale: Cho phép sử dụng `tokenizers==0.13.3` có sẵn wheel binary cho macOS arm64 (Apple Silicon), loại bỏ sự phụ thuộc vào trình biên dịch Rust khi build từ source trên Mac.

2. **`fewrel_5shot.py`:**
   - Gỡ bỏ dòng gán cứng thiết bị `os.environ["CUDA_VISIBLE_DEVICES"] = "7"`.
   - Bổ sung đọc các biến môi trường kiểm soát runtime:
     - `CONPL_DEVICE`: chọn `'mps'`, `'cuda'`, hoặc `'cpu'`.
     - `CONPL_SEQUENCES`: giới hạn số sequence chạy (1..6).
     - `CONPL_TASKS`: giới hạn số task trong mỗi sequence (1..8).
     - `CONPL_OUTPUT_DIR`: thư mục xuất checkpoint per-task (`checkpoint_latest.pt`) và metrics JSONL (`task_metrics.jsonl`).
     - `CONPL_PRETRAINED_MODEL`: cho phép trỏ tới thư mục checkpoint local hoặc HF cache thay vì tải lại.
   - Giữ nguyên 100% thuật toán gốc: loss function, consistency regularization, prototype calculation, exemplar selection, negative sampling.

---

## 2. Đặc tả Môi trường Cách ly (Isolated Virtualenv)

- **Đường dẫn venv:** `paper-reproductions/ConPL/.venv`
- **Hệ điều hành / Kiến trúc:** macOS (Darwin arm64, Apple Silicon M3 Pro)
- **Python version:** `Python 3.10.19` (CPython via `uv`)

### Danh mục Dependencies cốt lõi

| Package | Phiên bản | Ghi chú tương thích |
| :--- | :--- | :--- |
| `torch` | `2.2.2` | Hỗ trợ PyTorch MPS backend (`torch.backends.mps.is_available() == True`) |
| `numpy` | `1.23.5` | Giữ phiên bản `<1.24` để tránh lỗi ragged nested sequence khi tạo `np.asarray` |
| `tokenizers` | `0.13.3` | Bản prebuilt wheel ARM64 tương thích với patch dependency table |
| `huggingface-hub` | `0.10.1` | Tương thích với `transformers 4.22.0.dev0` đi kèm repo |
| `scikit-learn` | `1.5.2` | Phục vụ phân cụm K-Means cho Relation Clusters |
| `scipy` | `1.15.3` | Hỗ trợ tính toán khoa học |
| `wordninja` | `2.0.0` | Tách từ trong relation name |
| `six` | `1.17.0` | Thư viện compatibility |
| `tqdm` | `4.70.1` | Thanh tiến trình |

*Kết quả kiểm thử:* Import toàn bộ các module `transformers`, `encode.BERTSentenceEncoderPrompt`, `dataprocess.data_sampler_bert_prompt_deal_first_task`, `model.proto_softmax_layer_bert_prompt` thành công 100% không phát sinh cảnh báo lỗi.

---

## 3. Đặc thù Tính toán & Giới hạn đã xác định

1. **MPS Operator Fallback:**
   - Trong quá trình tính loss, `aten::multi_margin_loss` không có native kernel trên Metal Performance Shaders (MPS), PyTorch sẽ tự động fallback về CPU.
   - Tác động: Thời gian huấn luyện mỗi epoch của 1 task tốn khoảng ~18 giây trên Mac M3. Quá trình backward và update gradient vẫn diễn ra chính xác.
2. **Khởi tạo Seed & Mô hình:**
   - Mã gốc khởi tạo `encoderforbase` và `modelforbase` trước vòng lặp seed; trong vòng lặp mới gọi `set_seed(config, sequence_seed)` và `sampler.set_seed(...)`.
   - Để bảo toàn provenance của paper, run mã gốc giữ nguyên hành vi này; mọi điều chỉnh về thứ tự đặt seed (nếu có sau này) sẽ được đánh nhãn riêng là `seed-init-fix`.
