from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import os
import base64
import time
import shutil

app = Flask(__name__)
OLLAMA_URL = "http://localhost:11434"

# Cấu hình thư mục
BASE_DIR = os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload)
        res_json = response.json()
        b64_str = res_json.get('response', '')

        if "," in b64_str: b64_str = b64_str.split(",")[1]

        # 3. Lưu vào thư mục OUTPUTS
        output_filename = f"out_{int(time.time())}.png"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        image_bytes = base64.b64decode(b64_str)
        with open(output_filepath, "wb") as fh:
            fh.write(image_bytes)

        return jsonify({"success": True, "generated_image": output_filename})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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