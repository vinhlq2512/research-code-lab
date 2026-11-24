# Kế Hoạch Nghiên Cứu Bổ Sung: Track B – Data Scalability Study
## TAPTA: From Few-Shot to High-Resource Continual Relation Extraction

- **Mã văn kiện:** `SUPPLEMENTARY-RESEARCH-PLAN`
- **Ngày ban hành:** 2026-11-24 (Đồng bộ cùng W18)
- **Thuộc đề tài:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)
- **Định vị nghiên cứu:** **Track B (Supplementary Research)** — Nghiên cứu mở rộng phụ trợ, **tuyệt đối không thay thế Track A (Primary FCRE)**.
- **Trạng thái:** Active / On-Hold until Track A P0 Gate cleared

---

## 1. Mục Tiêu Nghiên Cứu Bổ Sung (Track B)

Trong khi trục nghiên cứu chính (**Track A**) tập trung giải quyết bài toán khắt khe **Rehearsal-Free Few-Shot Continual Relation Extraction (FCRE, 5-shot, $M=0$, No Task ID)**, nghiên cứu bổ sung (**Track B**) được thiết kế nhằm khảo sát:

> **Liệu phương pháp TAPTA (với Frozen BERT-base, Cây mồi nhắc phân cấp, Định tuyến Sub-Centroid phi tham số và Tổ hợp mềm tầng quyết định) có khả năng mở rộng quy mô dữ liệu (Data Scalability) lên môi trường High-Resource CRE hay không, trong khi vẫn bảo toàn tuyệt đối nguyên lý $M=0$ Rehearsal-Free?**

Mục tiêu cụ thể:
1. Đánh giá sự biến thiên của độ chính xác nhận dạng, độ chính xác định tuyến và mức độ quên lãng khi tăng kích thước tập huấn luyện từ 5 mẫu $\to$ 10 $\to$ 50 $\to$ 100 $\to$ Full (~420 mẫu/quan hệ).
2. Cung cấp bằng chứng thực nghiệm đối soát sòng phẳng với các công trình CRE kinh điển (như RP-CRE, EMAR, WAVE-CRE) trên cùng mặt bằng dữ liệu lớn, làm rõ ưu thế của thiết kế $M=0$ Rehearsal-Free so với các phương pháp Replay $M=10$.

---

## 2. Các Giả Thuyết & Câu Hỏi Nghiên Cứu Bổ Sung (Research Questions)

*Lưu ý: Đây là các giả thuyết khoa học cần kiểm chứng qua thực nghiệm trên GPU, không phải kết luận tiên nghiệm.*

* **RQ-B1 (Data Scalability Curve):** Hiệu năng nhận dạng tăng dần ($A_k$ và AIA) của TAPTA có tăng trưởng đơn điệu theo số lượng mẫu huấn luyện không? Tồn tại điểm uốn bão hòa (*diminishing returns*) ở mốc dữ liệu nào?
* **RQ-B2 (Prototype & Router Quality):** Việc có nhiều mẫu dữ liệu hơn có giúp các vector tâm con (*sub-centroids*) của từng quan hệ gom cụm chặt chẽ hơn, từ đó nâng độ chính xác định tuyến Top-1 và Top-2 Coverage của Task Router không?
* **RQ-B3 (Routing Gap Reduction):** Khoảng cách hiệu năng giữa chẩn đoán lý tưởng (Oracle Task-ID) và định tuyến thực tế (Task-Agnostic Decision Fusion) có được thu hẹp khi kích thước dữ liệu tăng không?
* **RQ-B4 (Catastrophic Forgetting Dynamics):** Khi các mồi nhắc nhánh $P^{\text{leaf}}_t$ được tối ưu hóa trên tập dữ liệu lớn hơn nhiều lần, khả năng kháng quên lãng của mô hình được củng cố hay bị suy giảm do hiện tượng quá khớp nội bộ tác vụ (*within-task overfitting*)?
* **RQ-B5 (Computational & Persistent Memory Scaling):** Thời gian huấn luyện, mức chiếm dụng bộ nhớ GPU (VRAM) tăng trưởng thế nào? Bộ đệm vector tĩnh (*persistent buffer*) có giữ vững mức cố định $\approx 1.03\text{ MB}$ không?

---

## 3. Ma Trận Thực Nghiệm Track B (Scalability Experiment Matrix)

Toàn bộ các thực nghiệm Track B tuân thủ nguyên tắc:
- Backbone: `bert-base-uncased` (109.5M tham số, đóng băng 100%).
- Bộ nhớ mẫu thô: $M = 0$ (hoàn toàn không replay câu văn cũ).
- Tập kiểm thử cố định: $8,000$ mẫu (100 mẫu/quan hệ $\times$ 80 quan hệ), lọc trùng toàn cục (`global_test_hashes`).
- Lấy mẫu lồng ghép xác định: $S_5 \subset S_{10} \subset S_{50} \subset S_{100} \subset S_{\text{Full}}$.

| Mã Exp | Nhãn Thiết Lập | Mẫu Train / Quan hệ | Tổng Mẫu Train | Tệp Cấu Hình | Độ Ưu Tiên |
|---|---|---|---|---|---|
| **DS-05** | 5-shot Reference | 5 | 400 | `configs/scalability/p4_fewrel_ds05_seed2021.yaml` | P2 (Baseline đối soát) |
| **DS-10** | 10-shot Low-Resource | 10 | 800 | `configs/scalability/p4_fewrel_ds10_seed2021.yaml` | P2 |
| **DS-50** | 50-shot Mid-Resource | 50 | 4,000 | `configs/scalability/p4_fewrel_ds50_seed2021.yaml` | P2 |
| **DS-100** | 100-shot High-Resource | 100 | 8,000 | `configs/scalability/p4_fewrel_ds100_seed2021.yaml` | P2 |
| **DS-FULL** | Full-Data CRE | ~420 | ~33,600 | `configs/scalability/p4_fewrel_ds_full_seed2021.yaml` | P2 |

---

## 4. Ngân Sách Tính Toán (Compute Budget) & Thời Gian Ước Tính

Đo đạc trên Google Colab GPU (NVIDIA Tesla T4 16GB VRAM, batch size 16):

| Thí nghiệm | Tổng Số Mẫu Huấn Luyện | Thời Gian Ước Tính / Task | Tổng Thời Gian Chạy (8 Tasks) | Chi Phí VRAM Ước Tính |
|---|---|---|---|---|
| **DS-05** | 400 mẫu | ~6 giây | ~1.5 phút | ~2.1 GB |
| **DS-10** | 800 mẫu | ~12 giây | ~2.5 phút | ~2.2 GB |
| **DS-50** | 4,000 mẫu | ~45 giây | ~8 phút | ~2.4 GB |
| **DS-100** | 8,000 mẫu | ~1.5 phút | ~15 phút | ~2.6 GB |
| **DS-FULL** | ~33,600 mẫu | ~5 phút | ~45 phút | ~3.2 GB |
| **Tổng Suite** | **46,800 lượt mẫu** | — | **~1.2 – 1.5 giờ GPU T4** | **< 4.0 GB** |

> [!NOTE]
> Ngân sách tính toán của toàn bộ Suite Track B là **dưới 1.5 giờ GPU**, hoàn toàn nằm trong hạn mức miễn phí hoặc Pro cơ bản của Google Colab, không gây nghẽn tài nguyên.

---

## 5. Quan Hệ Phụ Thuộc (Dependencies & Execution Gates)

Để đảm bảo tuyệt đối không làm trễ tiến độ của trục nghiên cứu chính:

```
[Track A Gate 1: W18 Architecture Freeze & Paper Draft Sec 2, 3, 4]
                             │
                             ▼
[Track A Gate 2: FewRel 5-seed Evaluation & TACRED Track A Execution] (P0)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Track A Ablations] (P1)          [Track B Scalability Run] (P2)
   (No-tree, Hard router)            (Chạy khi GPU rỗi / dôi dư)
```

**Quy tắc kích hoạt Track B:**
1. **Chỉ kích hoạt chạy GPU chính thức cho Track B sau khi:**
   - Hoàn thành viết xong khung Section 2, 3, 4 bản thảo bài báo (W18).
   - Đã chạy xong hoặc đang xếp hàng tự động cho các seeds chính của FewRel Track A (W19).
2. **Nếu gặp sự cố thiếu quota GPU Colab:** Dừng ngay lập tức Track B, ưu tiên 100% compute cho Track A.

---

## 6. Quản Trị Rủi Ro (Risk Management)

| Rủi Ro | Mức Độ | Biện Pháp Giảm Thiểu |
|---|---|---|
| **R1: Nhầm lẫn định vị bài báo** (Reviewer tưởng TAPTA là bài báo High-Resource CRE) | Cao | Khóa định vị: Abstract, Introduction và Conclusion chỉ tuyên bố đóng góp cho FCRE ($S=5, M=0$). Đưa toàn bộ kết quả Track B vào một tiểu mục riêng biệt: *Section 5.4: Supplementary Analysis — Data Scalability*. |
| **R2: Data Leakage trong Full-Data** | Nghiêm trọng | Đã triệt tiêu 100% bằng cơ chế `global_test_hashes` quét 80 quan hệ (đã xác thực qua pytest `tests/test_scalable_loader.py`). |
| **R3: OOM hoặc quá thời gian train Colab** | Thấp | Batch size cố định 16, BERT đóng băng 100% nên chỉ tính đạo hàm cho 3,072 tham số mồi nhắc/task. VRAM chỉ tốn ~3.2 GB. |

---

## 7. Tiêu Chuẩn Nghiệm Thu (Acceptance Criteria - AC)

* **AC-B1 (Zero Leakage):** Tuyệt đối 0 câu trùng lặp giữa tập train và test trên toàn bộ 80 quan hệ của cả 5 mức dữ liệu.
* **AC-B2 (Strict Invariance):** Kiến trúc mô hình, siêu tham số mồi nhắc ($L=8$), router ($\alpha=0.7$) và fusion ($K=2, \tau=0.2$) giống hệt 100% với Track A.
* **AC-B3 (Rehearsal-Free Storage):** Bộ nhớ lưu trữ bền vững (*persistent memory*) của cả 5 mức dữ liệu duy trì ở mức $\le 1.1\text{ MB}$.
* **AC-B4 (Artifact Completeness):** Mỗi cấu hình xuất đầy đủ 5 artifacts: `summary.json`, `performance_matrix.csv`, `resource_usage.json`, `dataset_manifest.json`, và `experiment.log`.

---

## 8. Định Nghĩa Hoàn Thành (Definition of Done - DoD)

Một thực nghiệm trong Track B được coi là hoàn tất khi:
1. Đã chạy thành công qua toàn bộ 8 stages trên Google Colab GPU (CUDA).
2. Ma trận $R_{i,j}$ ($8 \times 8$) được ghi nhận đầy đủ, không có giá trị NaN.
3. Tệp kết quả được nén và lưu trữ trong thư mục `runs/tapta_scalability/`.
4. Bảng số liệu đối sánh và biểu đồ đường cong tăng trưởng (Scalability Curve) được cập nhật vào báo cáo bổ trợ của đề tài.
