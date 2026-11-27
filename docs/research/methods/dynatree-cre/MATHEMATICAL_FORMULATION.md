# Công Thức Toán Học: DynaTree-CRE (Mathematical Formulation)

## 1. Phát Biểu Bài Toán (Continual Relation Extraction)

Cho chuỗi $K$ tác vụ tuần tự $\mathcal{T} = \{T_1, T_2, \dots, T_K\}$ với tập dữ liệu huấn luyện tại tác vụ $t$:
$$D_t = \{(x_i, h_i, e_i, y_i)\}_{i=1}^{N_t}$$
trong đó $x_i$ là câu chứa thực thể $(h_i, e_i)$ được đánh dấu bằng `[E1]...[/E1]` và `[E2]...[/E2]`, và $y_i \in \mathcal{R}_t$. Trong thiết lập Class-Incremental rời rạc:
$$\mathcal{R}_i \cap \mathcal{R}_j = \varnothing \quad (\forall i \ne j), \qquad \mathcal{R}_{\le t} = \bigcup_{j=1}^{t} \mathcal{R}_j$$
Tại thời điểm suy luận sau tác vụ $t$, mô hình dự đoán không có Task ID trên toàn bộ tập quan hệ đã quan sát:
$$\hat{y} = \arg\max_{r \in \mathcal{R}_{\le t}} p(r \mid x)$$

---

## 2. Stable Query Encoder & Stationary Centering

### 2.1. Mã hóa truy vấn bất biến
Cho backbone đóng băng $f_{\theta_0}$ (luôn ở chế độ `eval()`, không có dropout), vector truy vấn thô tại cặp vị trí marker đầu thực thể $([E1], [E2])$ là:
$$u(x) = [h_{[E1]}^{(0)}(x) ;\, h_{[E2]}^{(0)}(x)] \in \mathbb{R}^{2d}, \qquad q_{\text{raw}}(x) = \operatorname{Normalize}_2(u(x)) = \frac{u(x)}{\max(\|u(x)\|_2, \varepsilon)}$$

### 2.2. Hiệu chỉnh bất đẳng hướng cố định tại $T_1$ (T1-Centered Cosine Space)
Để khắc phục hiệu ứng hình nón (cone anisotropy) của BERT đóng băng mà vẫn giữ nguyên tính bất biến của prototype cũ qua mọi task:
1. Tại cuối $T_1$, tính vector trung bình neo bất biến từ tập prototype thô của $T_1$:
   $$c_r^{\text{raw}} = \operatorname{Normalize}_2\left(\frac{1}{N_r}\sum_{i: y_i = r} q_{\text{raw}}(x_i)\right), \qquad \mu_{\text{anchor}} = \frac{1}{|\mathcal{R}_1|} \sum_{r \in \mathcal{R}_1} c_r^{\text{raw}}$$
   Sau $T_1$, $\mu_{\text{anchor}}$ được **đóng băng vĩnh viễn**.
2. Ánh xạ bất kỳ vector $v \in \mathbb{R}^{2d}$ (query hoặc raw prototype) sang không gian tâm hóa tĩnh:
   $$\phi(v) = \operatorname{Normalize}_2\left(v - \mu_{\text{anchor}}\right)$$
   Khi đó $q(x) = \phi(q_{\text{raw}}(x))$ và $c_r = \phi(c_r^{\text{raw}})$ đảm bảo $c_r$ của mọi quan hệ cũ **không bao giờ thay đổi** ở các task $t > 1$.

---

## 3. Cây Ngữ Nghĩa Động (Dynamic Semantic Prompt Tree)

### 3.1. Prototype nút nội bộ (Cluster Prototype)
Với một cluster $v$ chứa tập lá $\text{Leaves}(v)$, prototype nút được tính bằng trung bình có trọng số đều ($w_r = 1$ ở MVP):
$$c_v = \operatorname{Normalize}_2\left(\sum_{r \in \text{Leaves}(v)} w_r c_r\right)$$
*(Lưu ý định nghĩa: $c_v$ là centroid chuẩn hóa của các relation centroids đã chuẩn hóa, không bằng centroid gộp của toàn bộ instance embeddings).*

### 3.2. Khởi tạo ở $T_1$ và Chèn gia tăng ở $T_t$ ($t \ge 2$)
- **Khoảng cách cosine tại $T_1$:** $d(a, b) = \operatorname{clip}(1 - \cos(c_a, c_b), 0, 2)$. Sử dụng hierarchical agglomerative clustering với **average linkage** và ngưỡng cắt $1 - \tau_{\text{ins}}$.
- **Chèn gia tăng tại $T_t$:** Với mỗi quan hệ mới $r \in \mathcal{R}_t$ (duyệt theo thứ tự tăng dần của `relation_id`):
  $$j^* = \arg\max_{j \in \mathcal{C}} \cos(c_r, c_j), \qquad s^* = \cos(c_r, c_{j^*})$$
  - Nếu $s^* \ge \tau_{\text{ins}}$: gắn $r$ làm con của cụm $j^*$ và cập nhật lại $c_{j^*}$.
  - Nếu $s^* < \tau_{\text{ins}}$: khởi tạo cụm mới $v_{\text{new}}$ chỉ chứa $r$ với $c_{v_{\text{new}}} = c_r$.

---

## 4. Residual Prompt Inheritance & Freeze Policy

Mỗi node lưu tensor residual $\mathbb{R}^{L \times d}$ ($L = 8$ mặc định). Prompt hiệu dụng cho quan hệ $r$:
$$P_r^{\text{eff}} = P_{\text{root}} + P_{\text{cluster}(r)} + \Delta P_r \in \mathbb{R}^{L \times d}$$
Số soft tokens ghép trước chuỗi đầu vào vẫn là $L$ (tính công bằng với concatenation $L = L_{\text{root}} + L_{\text{leaf}} = 8$ ở TAPTA `P4`).

---

## 5. Định Tuyến Mềm Phân Cấp (Task-Agnostic Hierarchical Soft Routing)

Với truy vấn $q(x)$:
1. **Xác suất cụm:**
   $$a_j(x) = \cos(q(x), c_j), \qquad \pi(j \mid x) = \frac{\exp(a_j(x) / \tau_c)}{\sum_{m \in \mathcal{C}} \exp(a_m(x) / \tau_c)}$$
2. **Xác suất lá có điều kiện trong cụm $j$:**
   $$b_r(x) = \cos(q(x), c_r), \qquad \pi(r \mid j, x) = \frac{\exp(b_r(x) / \tau_r)}{\sum_{u \in \text{Leaves}(j)} \exp(b_u(x) / \tau_r)}$$
3. **Xác suất lá toàn cục (tính trong log-space để ổn định số học):**
   $$\log \pi(r \mid x) = \log \pi(\text{cluster}(r) \mid x) + \log \pi(r \mid \text{cluster}(r), x), \qquad \sum_{r \in \mathcal{R}_{\le t}} \pi(r \mid x) = 1$$
4. **Lựa chọn Top-$K$ ($k_{\text{eff}} = \min(K, |\mathcal{R}_{\le t}|)$) và chuẩn hóa lại trọng số:**
   $$\alpha_r(x) = \frac{\pi(r \mid x)}{\sum_{u \in \text{TopK}(x)} \pi(u \mid x)} \quad (r \in \text{TopK}(x)), \qquad \beta_v(x) = \sum_{r \in \text{TopK}(x): \text{cluster}(r) = v} \alpha_r(x)$$
5. **Tổ hợp Prompt đầu vào:**
   $$P(x) = P_{\text{root}} + \sum_{v \in \mathcal{C}} \beta_v(x) P_v + \sum_{r \in \text{TopK}(x)} \alpha_r(x) \Delta P_r$$
   Lưu ý $\sum_v \beta_v(x) = \sum_{r \in \text{TopK}(x)} \alpha_r(x) = 1$, nên $P_{\text{root}}$ chỉ được cộng đúng 1 lần.

---

## 6. Bộ Phân Loại Toàn Cục & Hàm Mất Mát (MVP)

Đưa $P(x)$ và câu đã chèn marker qua encoder đóng băng để thu $z(x) = f_{\theta_0}(x; P(x)) \in \mathbb{R}^{2d}$.

1. **Expandable Cosine Classifier (Primary):**
   $$l_r(x) = s \cdot \frac{\langle z(x), w_r \rangle}{\max(\|z(x)\|_2 \|w_r\|_2, \varepsilon)} \quad (\forall r \in \mathcal{R}_{\le t})$$
   Hàm mất mát huấn luyện tại task $t$:
   $$\mathcal{L}_{\text{CE}} = -\frac{1}{|B|} \sum_{i \in B} \log \frac{\exp(l_{y_i}(x_i))}{\sum_{r \in \mathcal{R}_{\le t}} \exp(l_r(x_i))}$$
   Trong MVP, toàn bộ $w_r$ với $r \in \mathcal{R}_{<t}$ nằm ở buffer bất biến; chỉ $w_r$ với $r \in \mathcal{R}_t$ và các prompt mới được cập nhật.

2. **Prompted NCM Prototype Head (Control song song):**
   Sau khi hoàn tất train task $t$, tính nguyên mẫu biểu diễn $p_r = \operatorname{Normalize}_2(\frac{1}{N_r}\sum_{i: y_i = r} z(x_i))$ trên $D_t$ và dự đoán $\hat{y}_{\text{NCM}} = \arg\max_{r \in \mathcal{R}_{\le t}} \cos(z(x), p_r)$.

---

## 7. Công Thức Đánh Giá Chuẩn (Tái sử dụng từ `evaluation/metrics.py`)

Từ ma trận tam giác dưới $A_{t, j}$ ($0 \le j \le t \le K-1$):
- **Final Average Accuracy:** $\text{AA}_T = \frac{1}{K}\sum_{j=0}^{K-1} A_{K-1, j}$
- **Average Incremental Accuracy:** $\text{AIA} = \frac{1}{K}\sum_{t=0}^{K-1} \left(\frac{1}{t+1}\sum_{j=0}^t A_{t,j}\right)$
- **Average Forgetting:** $F = \frac{1}{K-1}\sum_{j=0}^{K-2}\left(\max_{l \in \{j,\dots,K-2\}} A_{l, j} - A_{K-1, j}\right)$
- **Backward Transfer:** $\text{BWT} = \frac{1}{K-1}\sum_{j=0}^{K-2}(A_{K-1, j} - A_{j, j})$
- **Routing Entropy:** $H(\pi(\cdot \mid x)) = -\sum_{r \in \mathcal{R}_{\le t}} \pi(r\mid x)\log(\pi(r\mid x) + \varepsilon)$
