import os
import cv2
from flask import abort
import numpy as np
import pickle
import base64
import json
from scipy.spatial import distance
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn 16MB chống lỗi 413

# ------------------- CONFIGURATION -------------------
BIN_SIZE = (8, 8, 8)             
NORMALIZATION = "l1"             
# ---------------------------------------------------

def load_data():
    with open("features.pkl", "rb") as f:
        return pickle.load(f)

_raw_database = load_data()

def normalize_hist(hist, norm_type="l2"):
    if norm_type == "l1":
        hist_sum = np.sum(hist)
        if hist_sum != 0:
            hist = hist / hist_sum
    else:
        hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L2)
    return hist.flatten()

processed_db = []
for item in _raw_database:
    img_path = item.get("path", "")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    # Cơ chế an toàn: Nếu không tìm thấy file ảnh gốc, dùng luôn histogram cũ
    if img is None:
        hist = item.get("histogram", np.zeros((512,), dtype=np.float32))
        processed_db.append({
            "path": img_path,
            "label": item.get("label", "unknown"),
            "histogram": hist,
            "hist_original": hist.copy()
        })
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, BIN_SIZE, [0, 180, 0, 256, 0, 256])
        hist = normalize_hist(hist, NORMALIZATION)
        processed_db.append({
            "path": img_path,
            "label": item.get("label", "unknown"),
            "histogram": hist,
            "hist_original": hist.copy()
        })

database = processed_db

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
            # CHUYỀN BIẾN NGẮN GỌN CHỐNG GÃY DÒNG CHO HTML
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
    # Chế độ mặc định của Flask (chỉ khởi động lại server, không tự refresh trình duyệt)
    # app.run(debug=True, port=5000)
    
    # Chế độ Live Reload (Tự động refresh trình duyệt khi code HTML/Python thay đổi)
    from livereload import Server
    app.debug = True
    server = Server(app.wsgi_app)
    
    # Theo dõi sự thay đổi của file index.html
    server.watch('templates/index.html')
    
    # Khởi chạy server ở cổng 5000
    server.serve(port=5000)