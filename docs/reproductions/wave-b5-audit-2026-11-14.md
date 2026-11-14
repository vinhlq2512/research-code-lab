# Audit Kỹ Thuật WAVE-CRE và WAVE++ cho Comparator B5

- **Ngày audit:** 2026-11-14
- **Branch:** `feat/b5-wave-smoke`
- **Mục tiêu:** Kiểm tra repo, commit, dependency, protocol, input data hash và blocker thực tế khi chạy trên Google Colab / GPU.

---

## 1. Thông tin Kho Mã Nguồn (Code Repositories)

| Thuộc tính | WAVE-CRE (mrshaw01) | WAVE++ (PiDinosauR2804) |
| :--- | :--- | :--- |
| **Repo URL** | `git@github.com:mrshaw01/AdaptivePromptingCRE.git` | `https://github.com/PiDinosauR2804/WAVE-CRE-PLUS-PLUS.git` |
| **Commit HEAD** | `5273b70` (Thu Dec 12 11:44:08 2024 +0700 `#0: Beautify`) | `7c9686a` (Tue Jul 15 13:56:30 2025 +0700 `update`) |
| **Trạng thái clone local** | `~/code/AdaptivePromptingCRE` | `~/code/WAVE-CRE-PLUS-PLUS` |
| **Thay đổi chưa commit** | Sửa `run.py` fallback CPU (local) | Sạch (clean clone) |
| **Dependencies chính** | `torch`, `transformers`, `gpytorch==1.6.0`, `linear_operator` | `torch`, `transformers`, `gpytorch==1.0`, `python-dotenv`, `wandb` |

---

## 2. Kiểm Tra Tính Toàn Vẹn & Hash Dữ Liệu

| Tệp dữ liệu | SHA-256 Checksum | Nhận xét |
| :--- | :--- | :--- |
| `FewRel/data_with_marker.json` | `2f781b172761ee1f2afb554e6a82c8e8fbd4ef17e8ad90c072c438a2500a062f` | Cả 2 repo dùng chung 100% |
| `FewRel/id2rel.json` | `157dcfe66976f72b2b2954629c73367c7659279bf4fc6a56551829f3475f2c4b` | Cả 2 repo dùng chung 100% |
| `TACRED/data_with_marker_tacred.json` (WAVE) | `9c8a1531b78b4db82e3f0643bbb663edc69081cab99ace3fe00b31173d3530c0` | Bản của WAVE-CRE |
| `TACRED/data_with_marker_tacred.json` (WAVE++) | `9855b4f5d528be7c4528b46d780f13382ab81059b96e46b6becbcea2fbbd6b2d` | WAVE++ đã cập nhật/tiền xử lý lại |
| `TACRED/id2rel_tacred.json` | `210e28cffc307fad96e40c7b81cb2da2f890459ea99268aa4bb465b3eb01f20d` | Cả 2 repo dùng chung 100% |

---

## 3. Phân Tích Protocol & Kiến Trúc Tác Giả

1. **Setting dữ liệu gốc:**
   - **FewRel:** 10 tasks $\times$ 8 relations = 80 relations. Mỗi relation có 420 train, 140 val, 140 test. Đây là **data-rich continual relation extraction**, hoàn toàn **KHÔNG PHẢI strict 5-shot**.
   - **TACRED:** 10 tasks $\times$ 4 relations = 40 relations (đã loại bỏ `no_relation`). Mẫu train tối đa 320/relation, test $\approx$ 40/relation.
2. **Inference & Routing:**
   - Không cung cấp Task ID lúc inference (Task-Incremental without Task Oracle / Class-Incremental).
   - Mô hình gồm: Encoder $\rightarrow$ First-stage Classifier (chọn Pool ID $\hat{t}$) $\rightarrow$ Prompted Encoder (với prompt trích từ Pool $\hat{t}$) $\rightarrow$ Second-stage Prompted Classifier.
   - Hàm `evaluate_strict_model` tính 4 accuracy:
     - `acc[0]`: Classifier stage 1 (không prompt).
     - `acc[1]`: Task ID / Pool ID classification accuracy.
     - `acc[2]`: Prompted accuracy với **Predicted Task ID** (ĐÂY LÀ METRIC CHÍNH CỦA B5).
     - `acc[3]`: Prompted accuracy với **Oracle Task ID** (Upper bound đối chứng).
3. **Bộ nhớ & Replay:**
   - Replay latent Gaussian qua SWAG / GMM (`replay_s_e_e = 100` hoặc `256`), không lưu raw text examples trong bộ nhớ chính ($M=0$ raw exemplar + latent parameter statistics).

---

## 4. Blocker Thực Tế & Lưu Ý Khi Chạy Trên Colab

1. **Khối lượng tính toán (Compute Blocker):**
   - Config tác giả cho 1 task: 30 epochs encoder + 25 epoch prompt pool + 250 epochs classifier + 200 epochs replay = **hơn 500 epochs / task**!
   - 10 task $\times$ 500 epochs trên CPU/Mac MPS sẽ mất nhiều ngày. **Bắt buộc chạy trên Colab GPU (T4/V100/A100)**.
2. **WandB Key hardcode trong WAVE++:**
   - Trong `WAVE-CRE-PLUS-PLUS/run.py` có hardcode API key wandb: `0806b2d5c00870a95f366d95c825d7680649abb7` và lệnh `wandb.login()`.
   - Khi chạy cần set `os.environ["WANDB_MODE"] = "disabled"` hoặc truyền flag tắt để tránh fail authen.
3. **Bare except trong evaluation:**
   - Trong `evaluate_strict_model` của cả hai repo, đoạn trích xuất có `try... except: sampled -= len(labels); continue`. Cần theo dõi stdout để đảm bảo không bị drop mẫu lỗi âm thầm.
4. **GPyTorch Version:**
   - `gpytorch` cần cài đúng phiên bản tương thích với PyTorch trên Colab (thường `pip install gpytorch linear_operator`).

---

## 5. Kết Luận Bước Đi Cho B5

- **Bước 1 (W14):** Smoke 1 seed (seed 2021) trên Colab với cấu hình tác giả (Track B / data-rich) để xác nhận code chạy được từ đầu tới cuối và đối chiếu kết quả với file pickle của tác giả.
- **Bước 2 (Chuyển tiếp W15+):** Viết Dataset Sampler Adapter để đưa mô hình B5 (WAVE-CRE) về thẩm định trên **Protocol v1 Track A (FewRel 8×10 5-shot) và TACRED 5-shot**.
- **Nếu thất bại:** Kiểm tra repo của **EoE** (Ensemble-of-Experts) làm phương án thay thế B5.
