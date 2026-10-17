import os
import cv2
import numpy as np
import base64
import json
from scipy.spatial import distance
from flask import Flask, render_template, request, abort, send_file
from supabase import create_client, Client
from dotenv import load_dotenv

# Kích hoạt việc đọc file .env
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB chống lỗi 413

# ------------------- CONFIGURATION -------------------
BIN_SIZE = (8, 8, 8)             
NORMALIZATION = "l1"             

# ------------------- SUPABASE SETUP -------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Thiếu cấu hình Supabase! Vui lòng kiểm tra lại file .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def normalize_hist(hist, norm_type="l2"):
    if norm_type == "l1":
        hist_sum = np.sum(hist)
        if hist_sum != 0:
            hist = hist / hist_sum
    else:
        hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)
    return hist.flatten()

def load_data_from_supabase():
    print("Đang tải dữ liệu từ Supabase... Vui lòng đợi!")
    # Lệnh select('*') lấy toàn bộ. limit(2000) để đảm bảo lấy đủ 1000 ảnh.
    response = supabase.table('cbir_features').select('*').limit(2000).execute()
    
    db_data = []
    for row in response.data:
        img_url = row["image_url"]
        
        # 1. Cắt URL để lấy nhãn chuẩn xác nhất (tránh lỗi nhãn cũ trong DB)
        url_parts = img_url.split('/')
        actual_label = url_parts[-2] if len(url_parts) >= 2 else row["label"]

        # Lấy mảng vector thô từ Database
        raw_hist = np.array(row["histogram"], dtype=np.float32)
        raw_orig = np.array(row["hist_original"], dtype=np.float32)
        
        # 2. FIX LỖI TOÁN HỌC: Ép chuẩn hóa L1 lại toàn bộ dữ liệu tải về RAM
        clean_hist = normalize_hist(raw_hist, NORMALIZATION)
        clean_orig = normalize_hist(raw_orig, NORMALIZATION)

        db_data.append({
            "path": img_url,
            "label": actual_label,
            "histogram": clean_hist,
            "hist_original": clean_orig
        })
    print(f"Tải thành công {len(db_data)} đặc trưng ảnh từ Database!")
    return db_data

# Load dữ liệu vào RAM ngay khi khởi động server
database = load_data_from_supabase()
# ---------------------------------------------------

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

def auto_predict_label(query_hist, database):
    best_label = "dinosaurs"
    min_dist = float('inf')
    for item in database:
        db_hist = item["histogram"]
        dist = chi2_distance(query_hist, db_hist)
        if dist < min_dist:
            min_dist = dist
            best_label = item["label"]
    return best_label

# Route này giữ lại phòng trường hợp load ảnh cục bộ
@app.route('/img/<path:filepath>')
def serve_img(filepath):
    decoded_path = filepath.replace('/', os.sep)
    if not os.path.isabs(decoded_path):
        decoded_path = os.path.abspath(decoded_path)
    if not os.path.isfile(decoded_path):
        abort(404)
    return send_file(decoded_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    labels = ["africans", "beaches", "buildings", "buses", "dinosaurs", "elephants", "flowers", "food", "horses", "mountains"]
    
    if request.method == 'POST':
        file = request.files.get('file')
        hidden_b64 = request.form.get('hidden_b64')
        hidden_hist = request.form.get('hidden_hist')
        hidden_hist_orig = request.form.get('hidden_hist_orig')
        
        if file and file.filename != '':
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            _, buffer = cv2.imencode('.jpg', image)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, BIN_SIZE, [0, 180, 0, 256, 0, 256])
            
            query_hist_orig = normalize_hist(hist, NORMALIZATION)
            query_hist = query_hist_orig.copy()
            query_label = auto_predict_label(query_hist, database)
            
        elif hidden_b64 and hidden_hist and hidden_hist_orig:
            img_b64 = hidden_b64
            query_hist = np.array(json.loads(hidden_hist), dtype=np.float32)
            query_hist_orig = np.array(json.loads(hidden_hist_orig), dtype=np.float32)
            
            selected_label = request.form.get('query_label', 'auto')
            if selected_label == 'auto':
                query_label = auto_predict_label(query_hist, database)
            else:
                query_label = selected_label
        else:
            return render_template('index.html', labels=labels, error="Vui lòng tải ảnh lên!")
            
        top_k = int(request.form.get('top_k', 10))
        
        res_euclid, res_manhat, res_cosine, res_chi2 = [], [], [], []
        
        # Quá trình so sánh cực nhanh vì DB trong RAM đã được "rửa sạch" ở hàm load_data_from_supabase
        for item in database:
            db_hist = item["histogram"]
            
            eu_dist = distance.euclidean(query_hist, db_hist)
            man_dist = distance.cityblock(query_hist, db_hist)
            cos_dist = distance.cosine(query_hist, db_hist)
            chi2_dist = chi2_distance(query_hist_orig, item["hist_original"])
            
            res_euclid.append({"dist": eu_dist, "path": item["path"], "label": item["label"]})
            res_manhat.append({"dist": man_dist, "path": item["path"], "label": item["label"]})
            res_cosine.append({"dist": cos_dist, "path": item["path"], "label": item["label"]})
            res_chi2.append({"dist": chi2_dist, "path": item["path"], "label": item["label"]})
            
        res_euclid = sorted(res_euclid, key=lambda x: x["dist"])[:top_k]
        res_manhat = sorted(res_manhat, key=lambda x: x["dist"])[:top_k]
        res_cosine = sorted(res_cosine, key=lambda x: x["dist"])[:top_k]
        res_chi2 = sorted(res_chi2, key=lambda x: x["dist"])[:top_k]
        
        precisions = {
            "Euclidean (L2)": (sum(1 for res in res_euclid if res['label'] == query_label) / top_k) * 100,
            "Manhattan (L1)": (sum(1 for res in res_manhat if res['label'] == query_label) / top_k) * 100,
            "Cosine (CS)": (sum(1 for res in res_cosine if res['label'] == query_label) / top_k) * 100,
            "Chi-square (χ²)": (sum(1 for res in res_chi2 if res['label'] == query_label) / top_k) * 100
        }
        
        sorted_algos = sorted(precisions.items(), key=lambda x: x[1], reverse=True)

        pros = {
            "Euclidean (L2)": "công thức đo khoảng cách hình học trực tiếp trên không gian phân bố màu sắc, nhạy bén với các sai lệch biên độ",
            "Manhattan (L1)": "công thức tính tổng chênh lệch tuyệt đối giúp đo lường ổn định và chịu lỗi tốt trước các điểm dị biệt",
            "Cosine (CS)": "nó bỏ qua tác động của độ sáng tổng thể, giúp đánh giá chính xác tỷ lệ cấu trúc phân bố giữa các mảng màu",
            "Chi-square (χ²)": "công thức kiểm định thống kê chuẩn hóa theo tần suất gốc rất nhạy bén, giúp bóc tách chi tiết độ lệch giữa các dải màu"
        }
        
        cons = {
            "Euclidean (L2)": "khoảng cách L2 dễ bị ảnh hưởng bởi độ sáng tổng thể và quá nhạy cảm với các biến đổi nhỏ cục bộ",
            "Manhattan (L1)": "nó vẫn chịu ảnh hưởng tuyến tính từ độ lệch cường độ sáng và chưa phản ánh tối ưu tỷ lệ cấu trúc",
            "Cosine (CS)": "nó bỏ qua biên độ tổng thể nên trong một số trường hợp không phân biệt được các dải màu có mật độ thưa thớt khác nhau",
            "Chi-square (χ²)": "nó phạt rất nặng các sai lệch nhỏ ở các bin có tần suất thấp, dễ gây nhiễu cục bộ và sụt giảm độ chính xác"
        }
        
        results = {
            "img_b64": img_b64,
            "query_label": query_label,
            "query_hist": query_hist.tolist(),
            "query_hist_orig": query_hist_orig.tolist(),
            "top_k": top_k,
            "euclid": res_euclid,
            "manhat": res_manhat,
            "cosine": res_cosine,
            "chi2": res_chi2,
            "prec_euclid": precisions["Euclidean (L2)"],
            "prec_manhat": precisions["Manhattan (L1)"],
            "prec_cosine": precisions["Cosine (CS)"],
            "prec_chi2": precisions["Chi-square (χ²)"],
            "sorted_algos": sorted_algos,
            "pros": pros,
            "cons": cons
        }
        return render_template('index.html', labels=labels, results=results)
    else:
        return render_template('index.html', labels=labels, results=None)

if __name__ == '__main__':
    # Chế độ Live Reload (Tự động refresh trình duyệt khi code HTML/Python thay đổi)
    from livereload import Server
    app.debug = True
    server = Server(app.wsgi_app)
    
    # Theo dõi sự thay đổi của file index.html
    server.watch('templates/index.html')
    
    # Khởi chạy server ở cổng 5000
    server.serve(port=5000)