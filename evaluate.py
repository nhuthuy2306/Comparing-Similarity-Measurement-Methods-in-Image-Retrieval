import os
import numpy as np
from scipy.spatial import distance
from dotenv import load_dotenv
from supabase import create_client, Client

# ================= CẤU HÌNH HỆ THỐNG =================
NORMALIZATION = "l1"
TOP_K = 10

# Kết nối Supabase
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Lỗi: Không tìm thấy file .env")
    exit()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= CÁC HÀM TÍNH TOÁN =================
def normalize_hist(hist, norm_type="l2"):
    if norm_type == "l1":
        hist_sum = np.sum(hist)
        if hist_sum != 0:
            hist = hist / hist_sum
    else:
        import cv2
        hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)
    return hist.flatten()

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

def load_data():
    print("Đang tải dữ liệu từ Supabase để đánh giá...")
    response = supabase.table('cbir_features').select('*').limit(2000).execute()
    db_data = []
    
    for row in response.data:
        img_url = row["image_url"]
        url_parts = img_url.split('/')
        actual_label = url_parts[-2] if len(url_parts) >= 2 else row["label"]
        
        raw_hist = np.array(row["histogram"], dtype=np.float32)
        raw_orig = np.array(row["hist_original"], dtype=np.float32)
        
        # Ép chuẩn hóa dữ liệu trên RAM
        clean_hist = normalize_hist(raw_hist, NORMALIZATION)
        clean_orig = normalize_hist(raw_orig, NORMALIZATION)
        
        db_data.append({
            "id": img_url,  # Dùng URL làm ID để phân biệt
            "label": actual_label,
            "histogram": clean_hist,
            "hist_original": clean_orig
        })
        
    print(f"Đã tải thành công {len(db_data)} ảnh. Bắt đầu đánh giá...")
    return db_data

# ================= HÀM ĐÁNH GIÁ CHÍNH =================
def evaluate():
    database = load_data()
    total_images = len(database)
    
    sum_prec_euclid = 0.0
    sum_prec_manhat = 0.0
    sum_prec_cosine = 0.0
    sum_prec_chi2 = 0.0
    
    print(f"\nTiến hành tính Precision@{TOP_K} cho toàn bộ {total_images} ảnh...")
    
    for i, query_item in enumerate(database):
        query_id = query_item["id"]
        query_label = query_item["label"]
        query_hist = query_item["histogram"]
        query_hist_orig = query_item["hist_original"]
        
        res_euclid, res_manhat, res_cosine, res_chi2 = [], [], [], []
        
        # So sánh 1 ảnh với 999 ảnh còn lại
        for db_item in database:
            if db_item["id"] == query_id:
                continue # Bỏ qua chính bức ảnh đang làm query
            
            db_hist = db_item["histogram"]
            
            eu_dist = distance.euclidean(query_hist, db_hist)
            man_dist = distance.cityblock(query_hist, db_hist)
            cos_dist = distance.cosine(query_hist, db_hist)
            chi2_dist = chi2_distance(query_hist_orig, db_item["hist_original"])
            
            res_euclid.append({"dist": eu_dist, "label": db_item["label"]})
            res_manhat.append({"dist": man_dist, "label": db_item["label"]})
            res_cosine.append({"dist": cos_dist, "label": db_item["label"]})
            res_chi2.append({"dist": chi2_dist, "label": db_item["label"]})
            
        # Lấy Top K (Top 10)
        top_euclid = sorted(res_euclid, key=lambda x: x["dist"])[:TOP_K]
        top_manhat = sorted(res_manhat, key=lambda x: x["dist"])[:TOP_K]
        top_cosine = sorted(res_cosine, key=lambda x: x["dist"])[:TOP_K]
        top_chi2 = sorted(res_chi2, key=lambda x: x["dist"])[:TOP_K]
        
        # Tính tỷ lệ chính xác (Precision) cho 1 ảnh truy vấn này
        prec_euclid = sum(1 for res in top_euclid if res['label'] == query_label) / TOP_K
        prec_manhat = sum(1 for res in top_manhat if res['label'] == query_label) / TOP_K
        prec_cosine = sum(1 for res in top_cosine if res['label'] == query_label) / TOP_K
        prec_chi2 = sum(1 for res in top_chi2 if res['label'] == query_label) / TOP_K
        
        # Cộng dồn vào tổng
        sum_prec_euclid += prec_euclid
        sum_prec_manhat += prec_manhat
        sum_prec_cosine += prec_cosine
        sum_prec_chi2 += prec_chi2
        
        if (i + 1) % 100 == 0:
            print(f" -> Đã xử lý {i + 1}/{total_images} ảnh...")
            
    # Tính trung bình (Mean Precision)
    mAP_euclid = (sum_prec_euclid / total_images) * 100
    mAP_manhat = (sum_prec_manhat / total_images) * 100
    mAP_cosine = (sum_prec_cosine / total_images) * 100
    mAP_chi2 = (sum_prec_chi2 / total_images) * 100
    
    print("\n" + "="*60)
    print(f" KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ (MEAN PRECISION@{TOP_K})")
    print(f" Tập dữ liệu: {total_images} ảnh")
    print("="*60)
    print(f" 1. Khoảng cách Euclidean (L2)  : {mAP_euclid:.2f}%")
    print(f" 2. Độ tương tự Cosine (CS)     : {mAP_cosine:.2f}%")
    print(f" 3. Khoảng cách Manhattan (L1)  : {mAP_manhat:.2f}%")
    print(f" 4. Khoảng cách Chi-square (χ²) : {mAP_chi2:.2f}%")
    print("="*60)

if __name__ == "__main__":
    evaluate()