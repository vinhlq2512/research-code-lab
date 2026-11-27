# M7.2 Specification — Dual-Space Routing (`DUAL_SPACE_ROUTING_SPEC.md`)

**Status:** ACTIVE (M7.2)  
**Parent Method:** `DynaTree-CRE` (within TAPTA Master's Thesis)

---

## 1. Động cơ khoa học
Kết quả Gate 5 cho thấy `t1_centered_cosine` và `raw_cosine` phục vụ hai mục tiêu xung đột:
1. **Tổ chức cây ngữ nghĩa (`Tree Space`):** `t1_centered_cosine` loại bỏ hiện tượng nón dị hướng của BERT, giúp phân cụm cân đối (`4` cụm, `0%` singleton trên FewRel; `3` cụm, `0%` singleton trên TACRED) so với `raw_cosine` (`64.7%` singleton FewRel, `50%` singleton TACRED).
2. **Truy xuất phân biệt (`Retrieval Space`):** `raw_cosine` bảo toàn khoảng cách góc nguyên bản giữa các quan hệ (`60.08%` vs `58.61%` ở `DT-C0` FewRel) và không bị lệch tâm miền khi $T_1$ đồng nhất (Exp B).

---

## 2. Đặc tả Dual-Space
Tách biệt cấu hình thành hai tham số trong `PrototypeConfig`:
- `prototype.similarity_space`: Không gian xây dựng và mở rộng cây ngữ nghĩa (`t1_centered_cosine` hoặc `raw_cosine`).
- `prototype.retrieval_similarity_space`: Không gian truy xuất prototype tại router (`None` = mặc định dùng chung `similarity_space` để bảo toàn 100% tương thích ngược `DT-C0..DT-C4`; hoặc chỉ định rõ `raw_cosine` / `t1_centered_cosine`).

### Quy tắc toán học (Không lưu instance cũ)
- Với mọi quan hệ đã học $r \in R_{\le t}$, `SemanticPrototypeBank` lưu `raw_prototype_vector` $\bar{q}^{\text{raw}}_r$ (trung bình đã chuẩn hóa L2 của query thô).
- Không giả định trung bình của các vector đã chuẩn hóa sau biến đổi bằng biến đổi của trung bình thô trong mọi trường hợp; khi ở không gian `raw_cosine`, cluster retrieval prototype được dựng trực tiếp từ raw prototypes thành viên:
  $$\mu^{\text{raw}}_c = \text{Norm}_2\left(\frac{1}{|c|}\sum_{r \in c} \bar{q}^{\text{raw}}_r\right)$$
- Khi ở chế độ `Raw Tree + Centered Routing`, `anchor_mean` được tính một lần trên tập raw prototypes của $T_1$ (`t1_raw_anchor_mean`) và cố định vĩnh viễn để biến đổi đồng nhất query và prototypes lúc truy xuất.

---

## 3. Ma trận Thực nghiệm Đối chứng 2×2 (Colab GPU)
| ID | Tree Space | Tree Threshold | Retrieval Space | Mục đích kiểm chứng |
|---|---|---|---|---|
| **M7.2-A** | `raw_cosine` | `fixed=0.75` (và sensitivity `0.80/0.85`) | `raw_cosine` | Raw tree + raw routing |
| **M7.2-B** | `t1_centered_cosine` | `t1_quantile=0.70` | `t1_centered_cosine` | Centered tree + centered routing (MVP) |
| **M7.2-C** | `t1_centered_cosine` | `t1_quantile=0.70` | `raw_cosine` | **Centered tree + raw routing (Đề xuất M7.2)** |
| **M7.2-D** | `raw_cosine` | `fixed=0.75` | `t1_centered_cosine` | Raw tree + centered routing |
