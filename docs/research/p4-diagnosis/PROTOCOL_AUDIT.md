# TAPTA Experimental Protocol Audit & Integrity Report

**Date:** 2026-11-25  
**Auditor:** Senior AI Research Engineer (Antigravity)  
**Project:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Status:** COMPLETED & VERIFIED  

---

## 1. Executive Summary

Bản kiểm toán này rà soát toàn diện tính toàn vẹn của dữ liệu thực nghiệm, tính hợp lệ của giao thức Continual Learning (Protocol v1 Track A), ranh giới không lưu mẫu thô ($M=0$), và tính nhất quán của các độ đo hiệu năng được sử dụng trong đánh giá phương pháp đề xuất **P4 (Hierarchical Prompt Tree & Decision Fusion)** trên hai tập dữ liệu benchmark **FewRel** và **TACRED**.

---

## 2. Dataset Integrity Audit

### 2.1. FewRel Track A Verification
| Tiêu chí | Đặc tả quy định | Kết quả kiểm toán thực tế | Trạng thái |
|---|---|---|---|
| Số lượng quan hệ | 80 relations | 80 relations (`relation_name.txt`) | PASS |
| Phân chia Tasks | 8 tasks $\times$ 10 relations | 8 tasks, mỗi task 10 relations | PASS |
| Số lượng mẫu huấn luyện | Strict 5-shot per relation | 5 mẫu $\times$ 80 = 400 mẫu train | PASS |
| Áp dụng Task 1 | 5-shot bằng mọi task | Task 1 chính xác 50 mẫu (không dùng 100 mẫu như ConPL) | PASS |
| Số lượng mẫu kiểm thử | 100-shot per relation | 100 mẫu $\times$ 80 = 8,000 mẫu test | PASS |
| Khử trùng rò rỉ dữ liệu (Leakage) | 0 rò rỉ train $\cap$ test | Global Hash Matching loại bỏ 23 mẫu trùng lặp | PASS |
| Mã hóa thực thể (Entity Markers) | `[E1]...[/E1]`, `[E2]...[/E2]` | Chèn trực tiếp vào từ vựng đặc biệt của BERT | PASS |
| Cắt ngắn độ dài (Truncation) | `max_seq_length = 128` | Entity marker offsets được tính lại chính xác sau khi prepend prompt | PASS |
| Trật tự Tasks (Task Order) | Seed 2021 `conpl_seed_shuffle` | Tạo bảng hoán vị cố định, tái lập 100% | PASS |

### 2.2. TACRED Track A Verification
| Tiêu chí | Đặc tả quy định | Kết quả kiểm toán thực tế | Trạng thái |
|---|---|---|---|
| Số lượng quan hệ | 40 positive relations | 40 relations (`relation_name.txt`) | PASS |
| Xử lý `no_relation` | Loại bỏ nhãn âm tính | Hoàn toàn vắng mặt trong tập dữ liệu Protocol v1 Track A | PASS |
| Phân chia Tasks | 8 tasks $\times$ 5 relations | 8 tasks, mỗi task đúng 5 relations | PASS |
| Số lượng mẫu huấn luyện | Strict 5-shot per relation | 5 mẫu $\times$ 40 = 200 mẫu train | PASS |
| Số lượng mẫu kiểm thử | Lên đến 100-shot per relation | Tổng cộng 1,550 mẫu test phân bổ theo cụm | PASS |
| Cấu hình `num_classes` | Khớp 40 quan hệ thực | Config ghi 41, relation names có 40 (cần chuẩn hóa chính thức) | WARNING |

> [!WARNING] Cảnh báo cấu hình TACRED
> File cấu hình `configs/p4_hierarchical_prompt_tree_tacred_5shot_seed2021.yaml` ghi `num_classes: 41` và `total_relations: 41`, nhưng file `relation_name.txt` thực tế có 40 dòng. Mô hình khởi tạo ma trận classifier 41 hàng, với quan hệ thứ 41 không bao giờ xuất hiện trong dữ liệu train/test. Cần chuẩn hóa dứt điểm về 40 để tránh lãng phí chiều nhãn.

---

## 3. Rehearsal-Free Constraints ($M=0$) Audit

| Yêu cầu ràng buộc | Đặc tả Rehearsal-Free | Kiểm toán Source Code P4 | Kết luận |
|---|---|---|---|
| Raw Data Storage | $M=0$ (Tuyệt đối không lưu text, token IDs) | Không có buffer lưu văn bản cũ (`M=0`) | COMPLIANT |
| Generative Replay | Không sử dụng mô hình sinh phụ trợ (GPT/Diffusion) | Không gọi API sinh ngoài | COMPLIANT |
| Prototype Buffer | Chỉ lưu centroid vector trích xuất từ 5 mẫu train | Buffer kích thước `[num_classes, 1536]` | COMPLIANT (491 KB FewRel, 245 KB TACRED) |
| Router Centroid Buffer | Chỉ lưu vector trung bình không gian unprompted | Buffer `[8, 1536]` và subcentroid `[8, 10, 1536]` | COMPLIANT (< 150 KB) |
| Feature Buffer | Không lưu trữ mảng vector thô của các mẫu cũ | Chỉ lưu mean và diagonal variance | COMPLIANT |

---

## 4. Continual Learning Setting & Metric Formulation Audit

### 4.1. Class-Incremental Learning (CIL) vs. Task-IL
- **CIL Thật sự:** Tại thời điểm suy luận sau stage $t$, mô hình phải đưa ra dự đoán nhãn trong toàn bộ không gian quan hệ đã học từ Task 1 đến Task $t$ ($C_{seen} = t \times R$). Xác suất đoán ngẫu nhiên là $1 / C_{seen}$ ($1/80 = 1.25\%$ tại $T_8$ FewRel).
- **Ranh giới Task ID:** Ground-truth Task ID tuyệt đối không được cấp cho mô hình trong quá trình suy luận.
- **Phát hiện đối soát với ConPL/CPL:** Các baseline ConPL (AAAI 2023) và CPL (EMNLP 2024) trong mã nguồn chính thức chỉ đánh giá xếp hạng trên 10 quan hệ ứng viên được cung cấp sẵn (Candidate Ranking 10-way, xác suất ngẫu nhiên 10%), không phải Class-Incremental Learning toàn diện. Đây là nguồn gốc chính tạo ra ảo ảnh chênh lệch số liệu 85% vs 57%.

### 4.2. Công thức Độ đo (Metrics)
1. **Final Average Accuracy ($A_{last}$):**
   $$A_{last} = \frac{1}{T} \sum_{j=1}^T R_{T, j}$$
   *Đã đối soát: Khớp 100% với hàm `compute_continual_summary_metrics`.*
2. **Average Incremental Accuracy (AIA):**
   $$\text{AIA} = \frac{1}{T} \sum_{i=1}^T \left( \frac{1}{i} \sum_{j=1}^i R_{i, j} \right)$$
   *Đã đối soát: Phản ánh trung bình đường cong học tập qua 8 stages.*
3. **Average Forgetting ($F$):**
   $$F = \frac{1}{T-1} \sum_{j=1}^{T-1} \max_{k \in \{j \dots T-1\}} (R_{k, j} - R_{T, j})$$
   *Đã đối soát: Khớp chuẩn Continual Learning.*

---

## 5. Technical Risks & Vulnerabilities Identified

1. **Lỗi Checkpoint Serialization trong `CheckpointManager`:**
   Phương thức `save_checkpoint` gọi `hasattr(model, "get_state_dict")`. Tuy nhiên mô hình PyTorch chỉ có `state_dict()`. Hệ quả là checkpoint `model.pt` và router weights không được lưu xuống đĩa trong các lần chạy trước, chỉ có `stage_metadata.json`. Cần khắc phục ngay trong `cl_re_baselines/checkpoint.py`.
2. **Hardcoded Baseline References trong `run_p4.py`:**
   Các giá trị tham chiếu `ref_p0`, `ref_p1`, `ref_p2a` trong `run_p4.py` được gắn cứng số liệu của FewRel (52.825%, 70.1375%, 59.275%). Khi chạy TACRED, việc trừ số liệu tham chiếu FewRel làm cho chỉ số `oracle_gap_closed_pct` bị âm (-59.92%). Cần nạp số tham chiếu động theo dataset từ file cấu hình.
3. **Mismatched Prompt Space khi tính Decision Fusion:**
   Các prototype của Task $j$ được lưu trong không gian biểu diễn của prompt $P_j$. Khi suy luận với candidate prompt $k \ne j$, vector $h(x; P_k)$ được tính độ tương đồng trực tiếp với prototype $c_j(P_j)$. Hai không gian này có khoảng cách trôi dạt biểu diễn (representation discrepancy).
