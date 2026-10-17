import os
import cv2
import numpy as np
import time
from dotenv import load_dotenv
from supabase import create_client, Client

# ================= CẤU HÌNH HỆ THỐNG =================
BIN_SIZE = (8, 8, 8)
NORMALIZATION = "l1"

# Tự động đọc URL và Key từ file .env
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Lỗi: Không tìm thấy SUPABASE_URL hoặc SUPABASE_KEY trong file .env")
    exit()

# Khởi tạo kết nối với Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= HÀM TRÍCH XUẤT ĐẶC TRƯNG =================
def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None
        
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, BIN_SIZE, [0, 180, 0, 256, 0, 256])
    
    # Giữ lại một bản sao để tính Chi-square
    hist_orig = hist.copy()
    cv2.normalize(hist_orig, hist_orig, norm_type=cv2.NORM_L1)
    
    # Tính bản chuẩn hóa chính thức
    if NORMALIZATION == "l1":
        hist_sum = np.sum(hist)
        if hist_sum != 0:
            hist = hist / hist_sum
    else:
        cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
        
    return hist.flatten(), hist_orig.flatten()

# ================= HÀM CHÍNH =================
def main():
    dataset_path = "dataset"
    batch_data = []
    
    print("Bắt đầu trích xuất đặc trưng và chuẩn bị dữ liệu...")
    
    # 1. Đọc và trích xuất dữ liệu từ máy tính
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                
                # SỬA LỖI NHÃN: Lấy nhãn chính xác từ tên thư mục chứa bức ảnh đó
                label = os.path.basename(root)
                
                # Tạo URL Public trên Supabase Storage
                clean_path = image_path.replace("\\", "/")
                if clean_path.startswith("dataset/"):
                    file_path = clean_path.replace("dataset/", "", 1)
                else:
                    file_path = clean_path
                public_url = f"{SUPABASE_URL}/storage/v1/object/public/dataset/{file_path}"
                
                # Trích xuất vector đặc trưng
                hist_vector, hist_orig = extract_color_histogram(image_path)
                
                if hist_vector is not None:
                    batch_data.append({
                        "label": label,
                        "image_url": public_url,
                        "histogram": hist_vector.tolist(),
                        "hist_original": hist_orig.tolist()
                    })

    print(f"Đã trích xuất xong {len(batch_data)} ảnh. Bắt đầu đẩy lên Supabase...")
    
    # 2. Đẩy dữ liệu lên Supabase (Có cơ chế chống rớt mạng)
    batch_size = 25
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i : i + batch_size]
        success = False
        retries = 3  # Cho phép thử lại 3 lần nếu bị rớt mạng hoặc quá tải
        
        while not success and retries > 0:
            try:
                # Đẩy 25 dòng lên Database
                supabase.table('cbir_features').insert(batch).execute()
                print(f" -> Đã đẩy thành công {i + len(batch)}/{len(batch_data)}...")
                success = True
                time.sleep(1)  # Nghỉ 1 giây để máy chủ Supabase kịp xử lý
            except Exception as e:
                retries -= 1
                print(f" [!] Lỗi tại mốc {i}: {e}. Đang thử lại... (Còn {retries} lần)")
                time.sleep(3)  # Nghỉ 3 giây trước khi bắt đầu thử đẩy lại
                
        if not success:
            print(f"Đã hết số lần thử lại tại mốc {i}. Quá trình tải lên bị dừng.")
            break

    print("\nHOÀN TẤT! Toàn bộ dữ liệu sạch đã nằm trên Supabase.")

if __name__ == "__main__":
    main()