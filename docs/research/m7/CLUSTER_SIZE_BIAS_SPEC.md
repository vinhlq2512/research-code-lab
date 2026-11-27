# M7.3 Specification — Cluster-Size-Aware & Hybrid Routing (`CLUSTER_SIZE_BIAS_SPEC.md`)

**Status:** ACTIVE (M7.3)  
**Bằng chứng nền:** Trên `semantic_homogeneous` (4 cụm `{20, 43, 14, 3}`), `DT-C4` (`hierarchical_soft`) tụt `Route Top-1` `-16.56%` (`57.77% → 41.21%`) và `Cluster Top-1` `-29.62%` (`85.89% → 56.27%`) so với `DT-C2` (`flat_soft_topk`). Trên TACRED (`{4, 9, 28}`), `Route Top-1` chỉ đạt `27.85%`.

---

## 1. Phân tích toán học Cluster-Size Bias
Trong `hierarchical_soft`:
$$\log p(r \mid x) = \log p(c(r) \mid x) + \log \frac{\exp(s(q, \mu_r)/\tau_r)}{\sum_{r' \in c(r)} \exp(s(q, \mu_{r'})/\tau_r)}$$
Khi các leaf có độ tương đồng ngang nhau, mẫu số của $\log p(r \mid c, x)$ tỷ lệ thuận với $|c(r)|$ (số quan hệ trong cụm), gây bất lợi $-\log |c(r)|$ cho mọi quan hệ thuộc cụm lớn.

---

## 2. Ba biến thể đối chứng
### Variant A — `hierarchical_soft` (Đối chứng `DT-C4` hiện tại)
Giữ nguyên công thức log-softmax phân cấp không hiệu chỉnh.

### Variant B — `size_corrected_hierarchical`
$$s_{\text{size}}(r) = \log p(c(r) \mid x) + \log p(r \mid c(r), x) + \eta \log |c(r)|, \quad p(r \mid x) = \text{softmax}_r(s_{\text{size}}(r))$$
- **Tính chất bất biến 1:** Khi $\eta = 0$, phân phối $p(r \mid x)$ trùng khớp tuyệt đối với Variant A (`hierarchical_soft`).
- **Tính chất bất biến 2:** Khi $\eta = 1$ và các leaf + cluster có similarity bằng nhau tuyệt đối, mọi leaf ở cụm kích thước $N$ và cụm kích thước $M$ nhận **cùng xác suất bằng nhau**.

### Variant C — `hybrid_tree_routing`
$$s_{\text{hyb}}(r) = \text{sim}(q_{\text{ret}}, \mu_r) + \gamma \cdot \text{sim}(q_{\text{ret}}, \mu_{c(r)}), \quad p(r \mid x) = \text{softmax}_r\left(\frac{s_{\text{hyb}}(r)}{\tau_r}\right)$$
- **Tính chất bất biến 3:** Khi $\gamma = 0$, $p(r \mid x)$ và Top-K trùng khớp tuyệt đối với `flat_soft_topk` trong cùng không gian retrieval.
- **Tính chất bất biến 4:** Truy xuất ứng viên toàn cục (`global Top-K`), không hard-filter theo cụm, không dùng nhãn thật khi suy luận.

---

## 3. Chính sách chọn Hyperparameter ($\eta, \gamma$)
Do phát hiện `valid_0.txt == test_0.txt` trong `CFRLdata_10_100_10_5` (xem `RD-013`), **cấm** dùng `valid_0` hoặc `test_0` để chọn $\eta, \gamma$. Giá trị mặc định lý thuyết ($\eta = 1.0$, và kiểm chứng thêm $\gamma \in \{0.0, 0.25, 0.5\}$ chỉ dựa trên validation nội bộ từ `train_samples` hoặc giữ tập tham chiếu cố định trước khi đánh giá multi-seed `2022–2025`).
