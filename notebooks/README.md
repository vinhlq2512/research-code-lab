# Google Colab Notebooks — Continual Relation Extraction Lab

Thư mục này chứa các file Jupyter Notebook chuẩn hóa, được thiết kế để chạy trơn tru trên **Google Colab** (hỗ trợ tăng tốc GPU T4, V100 hoặc A100).

---

## 📚 Danh sách Notebooks

| Thực nghiệm | Vai trò | Notebook | Trực tiếp trên Colab |
| :--- | :---: | :--- | :---: |
| **Baseline B0: Sequential Fine-Tuning** | **Lower Bound** | [`b0_sequential_ft.ipynb`](b0_sequential_ft.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinhlq2512/research-code-lab/blob/main/notebooks/b0_sequential_ft.ipynb) |
| **Upper Bound: Joint Training (Multitask)** | **Upper Bound** | [`upper_bound_joint.ipynb`](upper_bound_joint.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinhlq2512/research-code-lab/blob/main/notebooks/upper_bound_joint.ipynb) |

---

## 🚀 Hướng dẫn chạy trên Google Colab

### Bước 1: Bật GPU Tăng tốc
1. Mở notebook trên Colab thông qua huy hiệu **Open In Colab** ở bảng trên (hoặc tải file `.ipynb` lên Colab).
2. Trên thanh menu Colab, chọn: **Runtime** -> **Change runtime type** (Thay đổi loại phần cứng).
3. Tại mục **Hardware accelerator**, chọn **T4 GPU** (hoặc A100 nếu dùng Colab Pro) và nhấn **Save**.

### Bước 2: Chạy các Cell theo thứ tự
1. **Cell 1 (GPU Preflight):** Kiểm tra `nvidia-smi` và xác nhận CUDA khả dụng.
2. **Cell 2 (Setup Repo):** Tự động clone codebase từ GitHub và cài đặt thư viện (`torch`, `transformers`, `matplotlib`, `pandas`).
3. **Cell 3 (FewRel Auto-Download):** Tự động tải 3 file dữ liệu gốc (`train_wiki.json`, `val_wiki.json`, `pid2name.json`) từ GitHub chính thức của FewRel vào `data/raw/fewrel/` mà không cần upload thủ công.
4. **Cell 4 (Execution):** Chạy thực nghiệm chính thức với mô hình BERT thật (`bert-base-uncased`).
5. **Cell 5 (Visualization):** Hiển thị trực tiếp các bảng thống kê, ma trận hiệu năng và các đường cong lãng quên (Forgetting curves) ngay trong cell kết quả.
6. **Cell 6 (Export Artifacts):** Tùy chọn nén toàn bộ kết quả, trọng số và đồ thị thành file `.zip` tải về máy tính cá nhân.

---

## 📊 Kết quả chuẩn mực kỳ vọng trên Colab (FewRel Track A 5-shot, seed 2021)

| Chỉ số khoa học | Baseline B0 (Lower Bound) | Upper Bound (Joint Training) |
| :--- | :---: | :---: |
| **Final Accuracy (AA)** | **8.76%** | **52.12%** |
| **Final Macro-F1** | **9.69%** | **48.62%** |
| **Average Catastrophic Forgetting ($AF$)** | **33.62%** | **0.00%** (N/A) |
| **Backward Transfer ($BWT$)** | **-33.62%** | — |
| **Khoảng cách Cận trên vs Cận dưới** | — | **+43.35%** |
