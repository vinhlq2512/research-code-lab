# RD-013 — Dual-Space Routing, Hybrid/Size-Corrected Routing & Validation Integrity

- **Ngày:** 2026-11-28
- **Trạng thái:** Accepted
- **Liên quan:** M7.0, M7.1, M7.2, M7.3, Section 10 Experimental Integrity

---

## 1. Phát hiện sai lệch Validation Split trong `CFRLdata_10_100_10_5`
Kiểm tra bằng `cmp -s` xác nhận:
`data/protocol_v1/fewrel/CFRLdata_10_100_10_5/valid_0.txt` **trùng khớp hoàn toàn từng byte** với `test_0.txt`.
- **Quyết định:** Tuyệt đối **không** dùng `valid_0` của `CFRLdata_10_100_10_5` để chọn hyperparameter (`eta`, `gamma`, `lambda`, `K`), vì điều này sẽ tương đương rò rỉ tập test.
- **Cơ chế thay thế cho M7:**
  1. Seed `2021` được ghi nhận chính thức là **development evidence** (không được chọn cấu hình rồi báo cáo seed 2021 như kết quả không thiên lệch).
  2. Khi cần sweep hyperparameter ($\eta, \gamma, \lambda, K$), chỉ sử dụng validation tách từ `train_samples` (ví dụ 4-train / 1-val nội bộ theo relation) hoặc cấu hình cố định theo nguyên lý toán học ($\eta = 1.0$, $\gamma = 0.25 / 0.5$), chốt cấu hình trước khi đánh giá trên các seed mới (`2022, 2023, 2024, 2025`).
  3. Không sửa đổi `train_0` / `test_0` gốc của `ProtocolV1Dataset`.

## 2. Quyết định tách Dual-Space & Giữ PromptedNCM làm Classifier chính
- `similarity_space` kiểm soát cấu trúc cây (`t1_centered_cosine` giữ cây không sụp đổ thành singletons).
- `retrieval_similarity_space` (`raw_cosine` trong M7.2) kiểm soát truy xuất ứng viên để khôi phục độ chính xác `DT-C0 Raw` (`60.08%`) mà vẫn tận dụng cây cân đối.
- `PromptedNCM` là classifier chính cho M7; `ExpandableCosineClassifier` giữ nguyên để đối chứng.
