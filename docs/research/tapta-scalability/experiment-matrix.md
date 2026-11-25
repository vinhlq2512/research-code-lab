# Ma Trận Thí Nghiệm & Lộ Trình Thực Nghiệm (Experiment Matrix)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Tài liệu:** `docs/research/tapta-scalability/experiment-matrix.md`

---

## 1. Ma Trận Thí Nghiệm Mở Rộng Dữ Liệu Chính (Main Scalability Matrix)

| Exp ID | Mô hình | Cỡ mẫu/quan hệ | Mẫu/task | Tổng mẫu train | Bộ nhớ thô ($M$) | Cơ chế Inference | Mục tiêu đo lường |
|:---:|---|---:|---:|---:|:---:|---|---|
| **DS-05** | TAPTA P4 | 5 | 50 | 400 | 0 | Task-Agnostic (Soft Fusion) | Baseline Few-Shot (Đối chứng E012) |
| **DS-10** | TAPTA P4 | 10 | 100 | 800 | 0 | Task-Agnostic (Soft Fusion) | Đo đạc bước nhảy hiệu năng $\times 2$ data |
| **DS-50** | TAPTA P4 | 50 | 500 | 4,000 | 0 | Task-Agnostic (Soft Fusion) | Đo đạc vùng trung chuyển Medium-Resource |
| **DS-100** | TAPTA P4 | 100 | 1,000 | 8,000 | 0 | Task-Agnostic (Soft Fusion) | Đo đạc điểm tiệm cận độ bão hòa router |
| **DS-FULL**| TAPTA P4 | Full (~420) | ~4,200 | ~33,600 | 0 | Task-Agnostic (Soft Fusion) | Đỉnh cao High-Resource (Đối đầu RP-CRE) |

---

## 2. Ma Trận Chẩn Đoán Bộ Định Tuyến (Routing Diagnostic Matrix)

Chạy trên cùng checkpoint của các thí nghiệm trên để đo đạc `Routing Gap`:
$$\text{Routing Gap} = A_8^{\text{Oracle}} - A_8^{\text{Task-Agnostic}}$$

| Exp ID | Mô hình | Cỡ mẫu/quan hệ | Cơ chế Inference | Checkpoint nguồn | Mục tiêu nghiên cứu |
|:---:|---|---:|---|---|---|
| **RD-05** | TAPTA P4 | 5 | Oracle Task ID | Checkpoint của DS-05 | Đo tổn thất định tuyến ở 5-shot |
| **RD-10** | TAPTA P4 | 10 | Oracle Task ID | Checkpoint của DS-10 | Đo tổn thất định tuyến ở 10-shot |
| **RD-50** | TAPTA P4 | 50 | Oracle Task ID | Checkpoint của DS-50 | Đo tổn thất định tuyến ở 50-shot |
| **RD-100** | TAPTA P4 | 100 | Oracle Task ID | Checkpoint của DS-100 | Đo tổn thất định tuyến ở 100-shot |
| **RD-FULL**| TAPTA P4 | Full (~420) | Oracle Task ID | Checkpoint của DS-FULL | Đo tổn thất định tuyến ở Full-Data |

---

## 3. Ma Trận Nghiên Cứu Cắt Bỏ Thành Phần (Ablation Matrix)

Tách biệt đóng góp của việc mở rộng dữ liệu vào **Chất lượng Nguyên mẫu (Prototype Estimation)** so với **Tối ưu hóa Mồi nhắc (Prompt Tuning)**:

| Exp ID | Nguồn dữ liệu tính Prototype | Nguồn dữ liệu train Prompt | Mục tiêu phân tích |
|:---:|---:|---:|---|
| **AB-01** | 5-shot | 5-shot | Đối chứng nguyên bản (DS-05) |
| **AB-02** | Full-Data (~420) | 5-shot | Tách riêng giá trị của việc làm giàu Prototype/Router |
| **AB-03** | 5-shot | Full-Data (~420) | Tách riêng giá trị của việc học Prompt với nhiều dữ liệu |
| **AB-04** | Full-Data (~420) | Full-Data (~420) | Tác động tương hỗ kết hợp (DS-FULL) |

---

## 4. Ma Trận Đối So sánh với Các Baselines Tiêu Biểu

| Mô hình | 5-shot ($A_8$) | Full-Data ($A_8$) | Ngân sách Memory ($M$) | Trạng thái triển khai |
|---|---:|---:|:---:|:---:|
| **Sequential FT (B0)** | 15.65% | 22.40% *(ước tính)* | $M=0$ | Đã có code trong repo |
| **Shared Prompt (P0)** | 52.83% | TBD | $M=0$ | Đã chạy E008 |
| **RP-CRE (ACL 2021)** | N/A *(không tối ưu few-shot)* | ~72.50% *(Paper)* | **$M=10$ raw (800 câu)** | Đã kiểm chứng paper |
| **EMAR (EMNLP 2020)** | N/A | ~68.40% *(Paper)* | **$M=10$ raw (800 câu)** | Đã kiểm chứng paper |
| **WAVE-CRE (ACL 2023)** | N/A | ~70.20% *(Paper)* | $M=0$ *(Gaussian stats)* | Đã kiểm chứng repo |
| **ConPL (AAAI 2023)** | 58.20% *(local)* / 66.5% *(Paper)* | N/A | **$M=1$ raw (80 câu)** | Đã kiểm chứng repo |
| **TAPTA P4 (Ours)** | **57.58% - 58.80%** | **TBD (Dự kiến đột phá)** | **$M=0$ Tuyệt đối** | **Mục tiêu nghiên cứu** |
