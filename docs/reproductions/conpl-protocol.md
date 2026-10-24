# Đặc tả Protocol & Kiểm định Dữ liệu ConPL (FewRel 5-shot)

- **Ngày xác minh:** 2026-10-13  
- **Mục tiêu:** Khóa toàn diện giao thức thực nghiệm (Experimental Protocol) và chứng thực tính toàn vẹn dữ liệu cho FewRel 10-way 5-shot theo paper ACL 2023.

---

## 1. Khóa Mã băm Dữ liệu (Input Data Provenance & Hashes)

Tất cả các tệp dữ liệu được khóa bằng SHA-256 trong thư mục `paper-reproductions/ConPL/data/fewrel/`:

| Tệp dữ liệu | SHA-256 Checksum | Vai trò |
| :--- | :--- | :--- |
| `relation_name.txt` | `e2bf1364fdfe927045457c4627eaec64bf7cf8a5d7bdd145491dd3b101a99fd0` | Danh sách 80 quan hệ chuẩn FewRel |
| `rel_index.npy` | `bf877a1b803646492899cf49abb456f7490ce38c92bdda46a341d0a6fcc563b8` | Chỉ mục relation ID |
| `rel_feature.npy` | `edecff6648585f39217bc05657d17085df2dc4f7c2880a5ef1c5c6c8e33dcb1c` | Đặc trưng ngữ nghĩa quan hệ phục vụ K-Means |
| `CFRLdata_10_100_10_5/rel_cluster_label_0.npy` | `5107e26201ce9a46370ac25d10c8dc6c640cc9221c9f7a12ece49751415d40d1` | Cụm quan hệ cho split 0 (8 cụm $\times$ 10 relations) |
| `CFRLdata_10_100_10_5/train_0.txt` | `af862bd183852ee5dd8be0fe1b85db0b1c520697f7b0f68a6f2e0ef25eca4dfe` | Tập huấn luyện thô |
| `CFRLdata_10_100_10_5/valid_0.txt` | `6d6f14ce646b986ba3a600c10826ac7e61a42be8e1885d250fdbf83e35db8727` | Tập validation thô |
| `CFRLdata_10_100_10_5/test_0.txt` | `6d6f14ce646b986ba3a600c10826ac7e61a42be8e1885d250fdbf83e35db8727` | Tập kiểm thử thô (trùng khớp hoàn toàn với validation) |

---

## 2. Kiểm chứng Sampler & Trật tự Huấn luyện (Task Sequence Verification)

Thực thi kiểm định trực tiếp trên `data_sampler_bert_prompt_deal_first_task`:

- **Tổng số quan hệ:** 81 (gồm 1 ID đệm `fill fill fill` ở vị trí 0 và 80 quan hệ thật của FewRel ở vị trí 1..80).
- **Trật tự cụm (Task order) cho Seed 100:** `[0, 4, 7, 6, 1, 3, 5, 2]`.
- **Cấu hình chia Task:** 8 tasks liên tục. Mỗi task chứa đúng **10 quan hệ mới**.
- **Số mẫu huấn luyện:** Hàm `_deal_first_task` cắt tỉa chính xác mỗi quan hệ về đúng **5 mẫu (5-shot)**. Do đó, **cả 8 task đều có đúng 50 mẫu huấn luyện / task** (10 relations $\times$ 5 shots).
- **Số mẫu kiểm thử:** Mỗi task kiểm thử trên 1000 mẫu (100 mẫu / relation $\times$ 10 relations). Tổng test set của 80 quan hệ là 8000 mẫu.

### Bảng Thống kê 8 Task thực tế của Seed 100

| Task ID | Cluster Gốc | Quan hệ mới | Lũy kế quan hệ đã thấy | Mẫu Train (mỗi rel) | Mẫu Test từng task đã thấy | Tổng test lũy kế |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T_1$** | Cluster 0 | 10 | 10 | 50 (5 mẫu/rel) | `[1000]` | 1000 |
| **$T_2$** | Cluster 4 | 10 | 20 | 50 (5 mẫu/rel) | `[1000, 1000]` | 2000 |
| **$T_3$** | Cluster 7 | 10 | 30 | 50 (5 mẫu/rel) | `[1000, 1000, 1000]` | 3000 |
| **$T_4$** | Cluster 6 | 10 | 40 | 50 (5 mẫu/rel) | `[1000, 1000, 1000, 1000]` | 4000 |
| **$T_5$** | Cluster 1 | 10 | 50 | 50 (5 mẫu/rel) | `[1000, 1000, 1000, 1000, 1000]` | 5000 |
| **$T_6$** | Cluster 3 | 10 | 60 | 50 (5 mẫu/rel) | `[1000, 1000, 1000, 1000, 1000, 1000]` | 6000 |
| **$T_7$** | Cluster 5 | 10 | 70 | 50 (5 mẫu/rel) | `[1000, 1000, 1000, 1000, 1000, 1000, 1000]` | 7000 |
| **$T_8$** | Cluster 2 | 10 | 80 | 50 (5 mẫu/rel) | `[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]` | 8000 |

---

## 3. Không gian Nhãn & Giao thức Đánh giá (Evaluation Protocol)

1. **Inference Label Space:**
   - Tại mỗi bước đánh giá sau task $t$, mô hình tính score dự đoán trên tập nhãn của **tất cả các relation đã học từ trước đến nay** (`seen_relations`, kích thước $10 \times t$).
   - Nhãn task ID **không** được cung cấp tại inference (Task-Incremental mà không có task oracle tại test time / Class-Incremental setting chuẩn).
2. **Khái niệm Accuracy:**
   - `accuracy_whole`: Đánh giá trên toàn bộ mẫu kiểm thử của các relation đã học tính đến thời điểm hiện tại ($\sum_{j=1}^t \text{test}_j$). Đây chính là metric được công bố trong Table 1 của paper ConPL (ACL 2023).
   - `accuracy_average`: Trung bình cộng điểm accuracy riêng lẻ của từng task đã học ($\frac{1}{t}\sum_{j=1}^t A_{t,j}$).
3. **Exemplar Memory:**
   - Mỗi quan hệ lưu trữ đúng 1 exemplar và 1 prototype trong bộ nhớ memory (`first_task_memory_size = 1`, `rel_memory_size = 1`).
