# Sổ Đăng Ký Rủi Ro & Biện Pháp Kiểm Soát (Risk Register)
**Dự án:** Continual Relation Extraction with Task-Aware Prompt Adaptation (TAPTA)  
**Tài liệu:** `docs/research/tapta-scalability/risk-register.md`

---

## Bảng Quản Trị Rủi Ro Kỹ Thuật & Khoa Học

| Mã Rủi Ro | Phân Loại | Mô Tả Rủi Ro | Mức Độ | Khả Năng | Biện Pháp Kiểm Soát & Phòng Ngừa | Trạng Thái |
|:---:|:---:|---|:---:|:---:|---|:---:|
| **RSK-01** | Khoa học | **Data Leakage giữa Train và Test khi mở rộng pool** | Rất cao | Trung bình | Xây dựng hàm kiểm tra băm nội dung token chuẩn hóa; loại trừ 100% mẫu test set 8,000 câu ra khỏi train pool. Thêm assert trong unit tests. | ĐÃ CÓ GIẢI PHÁP |
| **RSK-02** | Kỹ thuật | **Colab Timeout khi huấn luyện Full-Data (33.6K mẫu)** | Trung bình | Cao | Tăng batch size lên 32/64; lưu checkpoint sau mỗi task; xây dựng script có khả năng tự động resume từ task gần nhất. | ĐÃ CÓ GIẢI PHÁP |
| **RSK-03** | Phương pháp | **Nhiễu mẫu (Sampling variance) làm sai lệch kết luận** | Cao | Cao | Áp dụng bắt buộc cơ chế **Nested Sampling**: Tập mẫu của cấp thấp hơn luôn là tập con của cấp cao hơn với cùng seed. | ĐÃ CÓ GIẢI PHÁP |
| **RSK-04** | Kỹ thuật | **Phá vỡ tính tương thích ngược với Track A 5-shot cũ** | Nghiêm trọng | Thấp | Không sửa đè `protocol_v1_loader.py` mà tách thành `scalable_fewrel_loader.py`. Viết regression test đảm bảo kết quả 5-shot không đổi. | ĐÃ CÓ GIẢI PHÁP |
| **RSK-05** | Khoa học | **Mâu thuẫn định nghĩa ngân sách bộ nhớ ($M$) khi so sánh** | Cao | Thấp | Phân biệt rõ rệt cột $M$ trong mọi bảng: $M=0$ (Rehearsal-Free của TAPTA), $M=1$ (ConPL), $M=10$ (RP-CRE, EMAR). | ĐÃ CÓ GIẢI PHÁP |
