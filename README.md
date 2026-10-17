# So sánh các phương pháp đo độ tương đồng trong truy xuất ảnh

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Tổng quan

Dự án **Comparing Similarity Measurement Methods in Image Retrieval** triển khai một hệ thống **Content‑Based Image Retrieval (CBIR)**, cho phép so sánh ảnh dựa trên histogram màu và các chỉ số đo khoảng cách:

- Euclidean (L2)
- Manhattan (L1)
- Cosine similarity
- Chi‑square

Mục tiêu là đánh giá chỉ số nào mang lại độ chính xác truy xuất tốt nhất trên các tập ảnh chuẩn.

---

## Cài đặt

```bash
# Clone repository
git clone https://github.com/nhuthuy2306/Comparing-Similarity-Measurement-Methods-in-Image-Retrieval.git
cd Comparing-Similarity-Measurement-Methods-in-Image-Retrieval

# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
.\\venv\\Scripts\\activate   # Windows
# source venv/bin/activate      # macOS/Linux

# Cài đặt phụ thuộc
pip install -r requirements.txt
```

> **Lưu ý:** Dự án yêu cầu Python 3.8 hoặc mới hơn.

---

## Bộ dữ liệu

Thư mục `data/` chứa một bộ dữ liệu mẫu nhỏ, cấu trúc theo lớp:

```
data/
├─ class_1/
│   ├─ img001.jpg
│   └─ img002.jpg
└─ class_2/
    └─ img010.jpg
```

Bạn có thể thay thế bằng bất kỳ tập ảnh nào, chỉ cần sắp xếp theo cùng cấu trúc thư mục.

---

## Sử dụng

### 1️⃣ Ứng dụng web (demo)

```bash
python app.py
```
Mở trình duyệt tại `http://127.0.0.1:5000` → tải lên ảnh và xem top‑k kết quả cho mỗi chỉ số.

### 2️⃣ Đánh giá batch (CLI)

```bash
python evaluate.py
```
Kịch bản sẽ tính **Mean Average Precision (mAP)** cho mỗi chỉ số và in ra console.

#### Tham số CLI
| Tham số | Mô tả | Mặc định |
|---|---|---|
| `--data` | Thư mục chứa ảnh | `data/` |
| `--k` | Số ảnh top‑k | `10` |
| `--metrics` | Các chỉ số, ngăn cách bằng dấu phẩy | `euclidean,manhattan,cosine,chi-square` |

---

## Kết quả mẫu (dataset 1000 ảnh)

```
=> Method EUCLIDEAN: Average precision = 63.38%
=> Method MANHATTAN: Average precision = 66.53%
=> Method COSINE: Average precision = 63.38%
=> Method CHI‑SQUARE: Average precision = 72.80%
```

Chi‑square cho độ chính xác cao nhất, là lựa chọn ưu tiên cho bộ dữ liệu này.

---

## Kiến trúc dự án

```
├─ app.py               # Ứng dụng Flask
├─ evaluate.py          # Đánh giá batch
├─ search.py            # CLI tìm kiếm
├─ templates/
│   └─ index.html       # Giao diện web (glassmorphism, dark mode)
├─ static/
│   └─ style.css        # CSS hiện đại
├─ requirements.txt
└─ README.md
```

---

## Bảo mật & cải tiến mới

- Tải lên file ảnh được kiểm tra phần mở rộng (`.jpg`, `.jpeg`, `.png`).
- Đọc pickle an toàn với xác thực checksum (SHA‑256).
- Route `/img/<path>` được giới hạn trong thư mục `static` để tránh path traversal.
- Kích thước tải lên được giới hạn 16 MB.

---

## Đóng góp

1. Fork kho lưu trữ.
2. Tạo nhánh mới (`git checkout -b feat/<tên‑tính‑năng>`).
3. Đảm bảo code format bằng `black` và không có lỗi `flake8`.
4. Mở Pull Request với mô tả chi tiết.

---

## Giấy phép

Dự án này được cấp phép theo **MIT License** – xem file [LICENSE](LICENSE) để biết chi tiết.

---

*Được tạo vào ngày 2026‑10‑17.*
