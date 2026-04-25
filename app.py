from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import os
import base64
import binascii
import time

app = Flask(__name__)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_GENERATE_TIMEOUT = 300

# Cấu hình thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _extract_base64_candidate(res_json):
    """Ưu tiên schema mới `image`, fallback `images[0]`, rồi `response`."""
    if not isinstance(res_json, dict):
        return ""

    image_value = res_json.get("image", "")
    if image_value:
        return image_value

    images = res_json.get("images", [])
    if isinstance(images, list) and images:
        return images[0]

    return res_json.get("response", "")


def _normalize_base64_string(value):
    if not isinstance(value, str):
        return ""

    cleaned = value.strip()
    if not cleaned:
        return ""

    # Hỗ trợ data URL dạng data:image/png;base64,...
    if cleaned.lower().startswith("data:") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]

    # Loại khoảng trắng/newline để decode ổn định.
    return "".join(cleaned.split())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            flux_models = [m['name'] for m in models if 'flux2-klein' in m['name']]
            return jsonify({"success": True, "models": flux_models})
        return jsonify({"success": False, "models": []})
    except Exception as e:
        return jsonify({"success": False, "models": []})

@app.route('/api/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '')
    model = request.form.get('model', '')
    files = request.files.getlist('images')

    base64_images = []
    
    # 1. Lưu vào thư mục UPLOADS
    for file in files:
        if file.filename:
            filepath = os.path.join(UPLOAD_DIR, file.filename)
            file.save(filepath)
            with open(filepath, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                base64_images.append(encoded_string)

    # 2. Gọi Ollama
    payload = {"model": model, "prompt": prompt, "stream": False}
    if base64_images: payload["images"] = base64_images

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_GENERATE_TIMEOUT,
        )
        response.raise_for_status()
    except requests.Timeout:
        return jsonify({"success": False, "error": "Timeout khi gọi Ollama, model đang xử lý quá lâu."})
    except requests.RequestException as e:
        return jsonify({"success": False, "error": f"Lỗi kết nối/HTTP tới Ollama: {str(e)}"})

    try:
        res_json = response.json()
    except ValueError:
        return jsonify({"success": False, "error": "Ollama trả dữ liệu không phải JSON hợp lệ."})

    # 1. Bắt lỗi do Ollama trả về
    if isinstance(res_json, dict) and res_json.get('error'):
        return jsonify({"success": False, "error": f"Ollama báo lỗi: {res_json['error']}"})

    # 2. Trích xuất dữ liệu ảnh theo schema ưu tiên: image -> images[0] -> response
    b64_raw = _extract_base64_candidate(res_json)
    b64_str = _normalize_base64_string(b64_raw)
    if not b64_str:
        keys = list(res_json.keys()) if isinstance(res_json, dict) else []
        return jsonify({
            "success": False,
            "error": f"Ollama không trả dữ liệu ảnh hợp lệ. Keys nhận được: {keys}",
        })

    # 3. Decode base64 có validate để tránh ghi file rỗng/hỏng
    try:
        image_bytes = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"success": False, "error": "Dữ liệu ảnh base64 không hợp lệ từ Ollama."})

    if not image_bytes:
        return jsonify({"success": False, "error": "Dữ liệu ảnh rỗng từ Ollama."})

    # 4. Lưu vào thư mục OUTPUTS
    output_filename = f"out_{int(time.time())}.png"
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_filepath, "wb") as fh:
        fh.write(image_bytes)

    return jsonify({"success": True, "generated_image": output_filename})

@app.route('/api/clear-uploads', methods=['POST'])
def clear_uploads():
    """Xóa tất cả file trong thư mục uploads"""
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        return jsonify({"success": True, "message": "Đã dọn dẹp thư mục upload."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Route phục vụ ảnh từ 2 thư mục khác nhau
@app.route('/media/<folder>/<filename>')
def serve_media(folder, filename):
    target_dir = UPLOAD_DIR if folder == 'uploads' else OUTPUT_DIR
    return send_from_directory(target_dir, filename)

if __name__ == '__main__':
    # Sử dụng port 5001 để tránh xung đột
    app.run(host='0.0.0.0', port=5001, debug=True)
