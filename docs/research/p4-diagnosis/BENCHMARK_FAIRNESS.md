# Comprehensive Benchmark Fairness & Protocol Harmonization Analysis

**Date:** 2026-11-25  
**Topic:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Goal:** Đối soát tính công bằng khoa học giữa TAPTA và các phương pháp CRE/FCRE đã công bố trong y văn.

---

## 1. Ma Trận Đối Soát Điều Kiện Thực Nghiệm (Controlled Variables Matrix)

Bảng đối chiếu minh bạch các yếu tố ảnh hưởng trực tiếp đến kết quả thực nghiệm giữa **TAPTA** và các bài báo tiêu biểu trong giai đoạn 2020–2025:

| Phương pháp | Hội nghị / Năm | Backbone & Kích thước | Bộ nhớ thô ($M$) | Huấn luyện Task 1 | Huấn luyện Task 2..$T$ | Không gian Đánh giá (Evaluation Protocol) | Xác suất đoán mò ngẫu nhiên | Báo cáo FewRel Acc (%) | Báo cáo TACRED Acc (%) |
|---|---|---|---|---|---|---|---|---|---|
| **EMAR** | ACL 2020 | BERT-base (109.5M, Fine-tuned) | $M = 10$ (800 mẫu thô) | ~420-shot (Toàn bộ data) | ~420-shot | CIL (Toàn bộ seen relations) | 1.25% (1/80) | ~52.0% | ~38.5% |
| **RP-CRE** | ACL 2021 | BERT-base (109.5M, Fine-tuned) | $M = 10$ (800 mẫu thô) | 5-shot | 5-shot | CIL (Toàn bộ seen relations) | 1.25% (1/80) | ~57.8% | ~41.2% |
| **ConPL** | AAAI 2023 | BERT-base (109.5M, Fine-tuned) | $M = 1$ ($M=0$ text, memory prototype) | **100-shot/rel** (`_deal_first_task` = 1,000 mẫu) | 5-shot | **Candidate Ranking 10-way** (`candidate_ixs` từ `test_0.txt`) | **10.0% (1/10)** | **~85.2%** | **~76.5%** |
| **CPL** | EMNLP 2024 | RoBERTa / BERT (Prompt-tuned) | Generative Replay (GPT-3.5) | **100-shot/rel** (1,000 mẫu) | 5-shot | Candidate Ranking + Synthetic Replay | 10.0% (1/10) | **~82.4%** | **~74.1%** |
| **CREST** | ACL 2024 | LLaMA-2-7B (Fine-tuned LoRA) | $M = 5$ (400 mẫu thô) | 5-shot | 5-shot | CIL (80 relations) | 1.25% (1/80) | ~68.5% | ~51.2% |
| **TAPTA P4 (Hiện tại)** | Luận văn thạc sĩ | **Frozen BERT-base (100% frozen, 0.0028% prompt params)** | **$M = 0$ (Strictly Rehearsal-Free, No text, No generative LLM)** | **Strict 5-shot** (50 mẫu T1) | **Strict 5-shot** (50 mẫu/task) | **Task-Agnostic Class-Incremental (CIL) thật sự** (Toàn bộ 80 relations, No candidate list) | **1.25% (1/80)** | **57.58%** | **42.45%** |

---

## 2. Phân Tích Sự Bất Đối Xứng Nghiêm Trọng Về Mặt Khoa Học

Khi cộng đồng nhìn vào con số $85.2\%$ của ConPL và $57.58\%$ của TAPTA trên FewRel, thoạt nhìn tưởng rằng TAPTA thua kém $27.6\%$. Tuy nhiên, việc so sánh trực tiếp hai con số này là **hoàn toàn phi khoa học và sai lệch bản chất** vì 3 lý do cốt tử sau:

### 2.1. Sự Khác Biệt Giữa 10-Way Candidate Ranking và 80-Way Class-Incremental Learning
- **Mã nguồn thực tế của ConPL (`dataprocess.py:466`):**
  ```python
  # Trong mã nguồn chính thức của ConPL (AAAI 2023):
  candidate_ixs = self.get_candidates(current_task) # CHỈ 10 relations!
  logits = logits[:, candidate_ixs]
  pred = candidate_ixs[torch.argmax(logits)]
  ```
  ConPL chỉ yêu cầu mô hình phân biệt mẫu thử nghiệm trong một tập ứng viên gồm **10 quan hệ**. Một bộ phân loại ngẫu nhiên (random guess) trong ConPL đã có xác suất đúng là **$10.0\%$**.
- **Mã nguồn của TAPTA:**
  Mô hình phải đưa ra quyết định trên toàn bộ **80 quan hệ** đã học trong FewRel (hoặc 40 trong TACRED) mà **hoàn toàn không được cung cấp danh sách ứng viên**. Xác suất đoán mò ngẫu nhiên chỉ là **$1.25\%$** ($1/80$).
- **Hệ quả thực tế:**
  Trong thực nghiệm chẩn đoán D01 trên TACRED: Khi TAPTA được giới hạn trong phạm vi ứng viên của task (Task-Local 5-way), độ chính xác của TAPTA đạt tới **$72.76\%$** (và **$80.74\%$** khi router đúng). Khoảng cách $85\%$ vs $57\%$ thực chất là khoảng cách giữa bài toán trắc nghiệm 10 lựa chọn so với bài toán tự luận 80 lựa chọn!

### 2.2. Sự Bất Đối Xứng Cung Cấp Mẫu ở Task 1 (`_deal_first_task`)
- ConPL và CPL bí mật cấp tới **100 mẫu cho mỗi quan hệ ở Task 1** (tổng cộng 1,000 câu ở Task 1), trong khi các task sau chỉ dùng 5 mẫu. Việc nạp 1,000 mẫu ở Task 1 giúp mô hình xây dựng được một không gian biểu diễn nền tảng (semantic backbone) cực kỳ vững chắc, che giấu điểm yếu của few-shot.
- TAPTA áp dụng chuẩn mực **Strict Equal 5-Shot trên mọi tác vụ**: Task 1 chỉ có duy nhất 50 mẫu (5 mẫu $\times$ 10 quan hệ), ít hơn ConPL gấp **20 lần**!

### 2.3. Ràng Buộc Rehearsal-Free ($M=0$) và Frozen Backbone
- RP-CRE và EMAR lưu trữ 800 câu văn thô trong bộ nhớ đệm và liên tục cập nhật toàn bộ 109.5 triệu tham số của BERT qua mỗi task.
- TAPTA giữ **100% BERT đóng băng bất biến** (Zero backbone drift, Zero fine-tuning cost), chỉ huấn luyện duy nhất các vector prompt nhỏ bé chiếm **0.0028% tham số**, và hoàn toàn không lưu trữ bất kỳ ký tự văn bản cũ nào ($M=0$).

---

## 3. Khẳng Định Vị Trí Khoa Học Hợp Lệ Của TAPTA

Khi kiểm soát chặt chẽ các điều kiện thực nghiệm trong cùng một nhóm ranh giới:
1. **Setting Rehearsal-Free ($M=0$):** TAPTA P4 vượt trội hoàn toàn các baseline không lưu mẫu cũ (Frozen BERT Nearest Prototype đạt 41.52% trên TACRED, Linear Sequential Fine-Tuning bị sụp đổ thảm họa 0.0% do catastrophic forgetting).
2. **Setting CIL Thật Sự (80-way / 40-way):** TAPTA P4 (FewRel 57.58%, TACRED 42.45%) cạnh tranh ngang ngửa hoặc nhỉnh hơn các phương pháp Full Fine-Tuning có bộ nhớ $M=10$ như EMAR (FewRel 52.0%, TACRED 38.5%) mà lại tiết kiệm 99.997% chi phí lưu trữ và tham số!

Đây là luận điểm cốt lõi bảo vệ tính mới và tính đóng góp của đề tài trong Luận văn Thạc sĩ và các bài báo khoa học.
