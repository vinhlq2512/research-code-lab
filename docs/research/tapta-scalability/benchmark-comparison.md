# Báo Cáo Kiểm Chứng Benchmark & Phân Tích Tính Công Bằng (Benchmark Verification)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Chủ đề:** TAPTA – Data Scalability & High-Resource Continual Relation Extraction  
**Thời điểm hoàn thành:** 2026-11-24  
**Nhánh Git:** `feat/tapta-data-scalability`

---

## 1. Mục Đích & Nguyên Tắc So Sánh Công Bằng (Fairness Principles)

Để đảm bảo tính liêm chính khoa học và tránh ngụy biện so sánh chéo giữa các giao thức thực nghiệm không tương thích:
1. **Phân định rạch ròi hai sân chơi:**
   - **Full-Data / High-Resource CRE:** Khởi tạo từ EA-EMR, EMAR, RP-CRE, CRL, CRECL, WAVE-CRE. Huấn luyện với **420 mẫu / quan hệ** (tổng 33,600 mẫu train cho 80 quan hệ FewRel).
   - **Few-Shot CRE (FCRE):** Khởi tạo từ ConPL, ERDA, CPL. Huấn luyện với chỉ **5 hoặc 10 mẫu / quan hệ** (tổng 400 hoặc 800 mẫu train cho 80 quan hệ FewRel).
2. **Cảnh báo về "Initial Task Data-Richness":**
   - Một số công trình FCRE (như ConPL `deal_first_task`, CPL) âm thầm cung cấp **100 mẫu / quan hệ ở Task 1** và chỉ áp dụng 5 mẫu ở các Task 2..8, giúp Task 1 neo giữ biểu diễn mạnh hơn rất nhiều.
   - Giao thức chuẩn Protocol v1 Track A của dự án chúng ta tuân thủ **Strict Equal-Shot**: toàn bộ 8 tác vụ đều học đúng **5 mẫu / quan hệ** ($50 \times 8 = 400$ mẫu).
3. **Phân định ngân sách bộ nhớ ($M$):**
   - $M > 0$: Lưu mẫu thô văn bản cũ vào episodic memory ($M=1, 5, 10$ câu/relation).
   - $M = 0$: Rehearsal-Free tuyệt đối (không lưu bất kỳ câu văn thô nào của task cũ).

---

## 2. Bảng Đối Soát Chi Tiết 10 Công Trình Tiêu Biểu

| Công trình | Hội nghị | Setting | Số Task | Số Q.Hệ / Task | Mẫu Train Task 1 | Mẫu Train Task 2..K | Mẫu Test / Q.Hệ | Ngân sách Memory ($M$) | Backbone | Task ID lúc Test | Báo cáo Quên lãng ($F$) | Nguồn kiểm chứng |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **EA-EMR** | EMNLP 2019 | Full CRE | 10 | 8 | 420 | 420 | 140 | $M=1, 5, 10$ raw | BERT-base (FT) | No | Không | Paper Sec 4, Code |
| **EMAR** | EMNLP 2020 | Full CRE | 10 | 8 | 420 | 420 | 140 | $M=1, 5, 10$ raw | BERT-base (FT) | No | Không | Paper Sec 4, Code |
| **RP-CRE** | ACL 2021 | Full CRE | 8 hoặc 10 | 10 hoặc 8 | 420 | 420 | 140 | $M=10$ raw (800 câu) | BERT-base (FT) | No | Có ($F_T$) | Paper Sec 4, GitHub `fd2014cl/RP-CRE` |
| **CRL** | AAAI 2021 | Full CRE | 10 | 8 | 420 | 420 | 140 | $M=1, 5$ raw | BERT-base (FT) | No | Không | Paper Sec 5 |
| **CRECL** | COLING 2022 | Full CRE | 10 | 8 | 420 | 420 | 140 | $M=1, 5, 10$ raw | BERT-base (FT) | No | Không | Paper Sec 4 |
| **ERDA** | ACL 2022 | FCRE | 8 | 10 | 5 hoặc 10 | 5 hoặc 10 | 100 | $M=1, 5$ raw | BERT-base (FT) | No | Không | Paper Sec 4 |
| **ConPL** | AAAI 2023 | FCRE | 8 | 10 | 100 *(first-task)* | 5 *(sau)* | 100 | $M=1$ raw (80 câu) | BERT-base (FT) | No | Không | Paper Sec 4, `dataprocess.py` L388 |
| **WAVE-CRE** | ACL 2023 | Full CRE | 10 | 8 | 420 | 420 | 140 | $M=0$ raw *(Gaussian stats)* | BERT-base (Frozen) | No (Router) | Có ($F_T$) | Paper Sec 4, repo `WAVE-CRE` |
| **WAVE++** | ArXiv 2024 | Full CRE | 10 | 8 | 420 | 420 | 140 | $M=0$ raw *(Gaussian stats)* | BERT-base (Frozen) | No (Voting) | Có ($F_T$) | Paper Sec 4 |
| **CPL** | EMNLP 2024 | FCRE | 8 | 10 | 100 *(first-task)* | 5 *(sau)* | 100 | $M=0$ raw *(GPT replay)* | BERT-base (FT) | No | Có ($F_T$) | Paper Sec 4 |
| **TAPTA P4 (Ours - Few-Shot)** | 2026 | Strict FCRE | 8 | 10 | **5 (Strict)** | **5 (Strict)** | 100 | **$M=0$ Rehearsal-Free** | BERT-base (Frozen) | **No (Soft Router)** | **11.89% (Đỉnh cao)** | E012, `run_p4.py` |
| **TAPTA P4 (Ours - Scalability)** | 2026 | Scalability | 8 | 10 | **5, 10, 50, 100, Full** | **Equal Shots** | 100 | **$M=0$ Rehearsal-Free** | BERT-base (Frozen) | **No (Soft Router)** | **Nghiên cứu DS-05 $\to$ DS-FULL** | Nghiên cứu mở rộng hiện tại |

---

## 3. Các Điểm Khác Biệt Cần Lưu Ý Trong Phân Tích Học Thuật

1. **Sự Khác Biệt Về Kích Thước Tập Kiểm Thử (Test Set):**
   - Các bài toán Full CRE (RP-CRE, EMAR, WAVE-CRE) dùng tập test 140 mẫu/quan hệ ($140 \times 80 = 11,200$ câu test).
   - Các bài toán FCRE (ConPL, CPL, Protocol v1) dùng tập test 100 mẫu/quan hệ ($100 \times 80 = 8,000$ câu test).
   - Tỷ lệ phân bố nhãn trong cả hai tập test đều cân bằng tuyệt đối (Balanced Uniform Test Distribution).
2. **Cơ chế Replay trong Rehearsal-Free:**
   - **WAVE-CRE / WAVE++:** Không lưu text thô, nhưng lưu ma trận hiệp phương sai Gaussian của toàn bộ biểu diễn và mẫu truy vấn. Đòi hỏi GPU A100 (85GB RAM) khi số lượng quan hệ tăng lên.
   - **CPL:** Không lưu text thô, nhưng gọi GPT-3.5 API sinh ra hàng ngàn câu văn giả lập để replay.
   - **TAPTA (P4):** Hoàn toàn không lưu text, không replay Gaussian mẫu truy vấn, không gọi LLM sinh văn bản. Chỉ lưu **vector nguyên mẫu $p_c$** và **tâm định tuyến $\mu_t, \nu_{t,c}$** với tổng dung lượng toàn bộ 8 tác vụ chỉ là **~1.03 MB**.
3. **Giá trị của Data Scalability Study đối với TAPTA:**
   - Khi tăng kích thước mẫu huấn luyện từ 5 lên 10, 50, 100, và Full (420 mẫu), TAPTA là phương pháp DUY NHẤT kiểm chứng tính mở rộng tài nguyên trong điều kiện $M=0$ và BERT đóng băng 100%.
   - Cho phép so sánh trực tiếp hiệu năng với RP-CRE ($M=10$) và EMAR ($M=10$) ở chế độ Full-Data để chứng minh ưu thế của Cây mồi nhắc và Decision Fusion.
