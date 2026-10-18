# 🔍 SO SÁNH CÁC PHƯƠNG PHÁP ĐO ĐỘ TƯƠNG TỰ TRONG TRUY XUẤT ẢNH DỰA TRÊN  ĐẶC TRƯNG COLOR HISTOGRAM - Nhóm 14

## 📖 Giới thiệu Tổng quan
Dự án ứng dụng công nghệ **Truy xuất Ảnh dựa trên Nội dung** (Content‑Based Image Retrieval - CBIR), tập trung vào việc trích xuất đặc trưng màu sắc (Color Histogram) trong không gian màu **HSV**.

### Điểm nổi bật
- Đánh giá và so sánh **4 công thức đo khoảng cách/độ tương đồng** phổ biến nhất hiện nay:
  - **Khoảng cách Euclidean (L2)**
  - **Khoảng cách Manhattan (L1)**
  - **Độ tương tự Cosine (CS)**
  - **Khoảng cách Chi‑square ($\chi^2$)**
- Dữ liệu đặc trưng và hình ảnh được lưu trữ tập trung trên **Supabase (Cloud Database)**, giúp tối ưu kiến trúc và tăng tốc độ truy xuất.

## 👥 Thành viên Nhóm 14
- **Võ Hà Như Thủy** (MSSV: N23DCCN061) – Kiến trúc hệ thống, Backend Flask, Cloud Database & UI/UX.
- **Nguyễn Trần Ngọc Duyên** (MSSV: N23DCCN016) – Cài đặt thuật toán Histogram, các độ đo khoảng cách & Logic đánh giá.

## 🚀 Hướng dẫn Cài đặt & Chạy dự án
### 1. Yêu cầu hệ thống
- Python **3.8** trở lên.
- Git để **clone** dự án.

### 2. Cài đặt môi trường
Mở **Terminal/Command Prompt** và chạy các lệnh sau:
```bash
# Clone repository về máy
git clone https://github.com/nhuthuy2306/Comparing-Similarity-Measurement-Methods-in-Image-Retrieval.git
cd Comparing-Similarity-Measurement-Methods-in-Image-Retrieval

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```
### 3. Cấu hình Cơ sở dữ liệu Đám mây (Supabase)
Dự án sử dụng Supabase để lưu trữ vector đặc trưng. Bạn cần tạo một file tên là **.env** ở thư mục gốc của dự án (cùng cấp với `app.py`) và cấu hình như sau:
```
SUPABASE_URL="https://[PROJECT-ID].supabase.co"
SUPABASE_KEY="[YOUR-ANON-PUBLIC-KEY]"
```
> (Lưu ý: Liên hệ nhóm phát triển để nhận Key thực tế nếu bạn cần.)

### 4. Khởi chạy Ứng dụng
```bash
python app.py
```
Sau khi Terminal báo **thành công**, hãy mở trình duyệt web và truy cập vào địa chỉ: <http://127.0.0.1:5000>

## 📂 Cấu trúc Thư mục Chính
```
📦 Comparing-Similarity-Measurement-Methods-in-Image-Retrieval
 ┣ 📂 templates           # Giao diện web (index.html)
 ┣ 📜 app.py              # Server Flask chính, xử lý logic Web và kết nối DB
 ┣ 📜 extract_and_upload.py # Script tự động trích xuất vector và đẩy lên Supabase
 ┣ 📜 evaluate.py         # Script đánh giá các phương pháp đo độ tương đồng
 ┣ 📜 requirements.txt    # Danh sách thư viện Python
 ┣ 📜 .env                # File cấu hình bảo mật (không đưa lên Git)
 ┗ 📜 README.md           # Tài liệu hướng dẫn này
```

## 📊 Tính năng Nổi bật
- **Trực quan hoá Toán học**: Hiển thị công thức đo khoảng cách bằng chuẩn **$\LaTeX$** và biểu đồ phân bố độ chính xác.
- **Đánh giá Precision Top‑K**: Tự động tính toán tỷ lệ chính xác của từng thuật toán dựa trên nhãn (label) của tập dữ liệu.
- **In‑memory Cleansing**: Tự động ép chuẩn hoá (Normalization) dữ liệu nhiễu ngay trên RAM khi kết nối với Database để đảm bảo sai số toán học bằng **0**.
- **Xử lý Semantic Gap**: Minh họa rõ rệt nhược điểm của việc trích xuất màu sắc (Semantic Gap) thông qua các sai số thực tế giữa các thuật toán.

---

## 🤝 Đóng góp
1. Fork kho lưu trữ.
2. Tạo nhánh mới (`git checkout -b feat/<tên‑tính‑năng>`).
3. Đảm bảo code format bằng `black` và không có lỗi `flake8`.
4. Mở Pull Request với mô tả chi tiết.

---

## 📄 Giấy phép
Dự án này được cấp phép theo **MIT License** – xem file [LICENSE](LICENSE) để biết chi tiết.

---

*Được tạo vào ngày 2026‑10‑17.*
