from flask import Flask, request, jsonify, render_template, send_from_directory
import requests
import os
import base64
import binascii
import time
import json
import uuid

app = Flask(__name__)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_GENERATE_TIMEOUT = 300

VISION_PRIMARY_MODEL = "gemma4:e4b"
VISION_MAX_REFERENCE_IMAGES = 4

VISION_PRIMARY_TIMEOUT_BASE = 60
VISION_PRIMARY_TIMEOUT_PER_IMAGE = 30
VISION_PRIMARY_TIMEOUT_CAP = 180
VISION_CHAT_NUM_CTX = 8192
VISION_CHAT_NUM_PREDICT = 2048
VISION_CHAT_KEEP_ALIVE = 0

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


def _save_and_encode_uploads(files):
    """Lưu ảnh upload và trả về list base64 để gọi model vision."""
    encoded_images = []

    for file in files:
        if not file.filename:
            continue

        safe_name = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
        filepath = os.path.join(UPLOAD_DIR, safe_name)
        file.save(filepath)

        with open(filepath, "rb") as image_file:
            encoded_images.append(base64.b64encode(image_file.read()).decode("utf-8"))

    return encoded_images


def _strip_markdown_code_fence(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _parse_vision_json(content):
    cleaned = _strip_markdown_code_fence(content)

    # fallback: model trả thêm text thừa ngoài JSON
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("response không phải object JSON")
    return data


def _build_reference_prompt(analysis):
    """Chuyển JSON phân tích ảnh thành prompt tham chiếu dễ chỉnh sửa."""
    fields = [
        ("subject", "Chủ thể"),
        ("style", "Phong cách"),
        ("composition", "Bố cục"),
        ("lighting", "Ánh sáng"),
        ("color_palette", "Màu sắc"),
        ("camera", "Góc máy/ống kính"),
        ("mood", "Mood"),
        ("negative", "Tránh"),
    ]

    lines = []
    for key, label in fields:
        value = analysis.get(key, "")
        if isinstance(value, list):
            value = ", ".join([str(v).strip() for v in value if str(v).strip()])
        if value:
            lines.append(f"{label}: {str(value).strip()}")

    return "\n".join(lines).strip()


def _compute_primary_vision_timeout(image_count):
    return min(
        VISION_PRIMARY_TIMEOUT_BASE + VISION_PRIMARY_TIMEOUT_PER_IMAGE * image_count,
        VISION_PRIMARY_TIMEOUT_CAP,
    )


def _run_vision_analysis(model_name, base64_images, timeout_seconds, system_prompt):
    payload = {
        "model": model_name,
        "stream": False,
        "format": "json",
        "keep_alive": VISION_CHAT_KEEP_ALIVE,
        "options": {
            "num_ctx": VISION_CHAT_NUM_CTX,
            "num_predict": VISION_CHAT_NUM_PREDICT,
        },
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "Hãy phân tích ảnh tham chiếu và trả JSON theo schema đã yêu cầu.",
                "images": base64_images,
            },
        ],
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except requests.Timeout:
        return None, "timeout", f"Timeout sau {timeout_seconds}s khi gọi {model_name}."
    except requests.RequestException as e:
        return None, "request", f"Lỗi kết nối/HTTP với {model_name}: {str(e)}"

    try:
        res_json = response.json()
    except ValueError:
        return None, "parse", f"{model_name} trả dữ liệu không phải JSON hợp lệ."

    if isinstance(res_json, dict) and res_json.get("error"):
        return None, "model_error", f"{model_name} báo lỗi: {res_json['error']}"

    content = ""
    if isinstance(res_json, dict):
        message = res_json.get("message", {})
        if isinstance(message, dict):
            content = (message.get("content") or "").strip()
        if not content:
            content = (res_json.get("response") or "").strip()

    if not content:
        done_reason = res_json.get("done_reason") if isinstance(res_json, dict) else None
        reason_text = f", done_reason={done_reason}" if done_reason else ""
        return None, "parse", f"{model_name} không trả nội dung phân tích ảnh{reason_text}."

    try:
        analysis = _parse_vision_json(content)
    except (ValueError, json.JSONDecodeError) as e:
        return None, "parse", f"Không parse được JSON từ {model_name}: {str(e)}"

    reference_prompt = _build_reference_prompt(analysis)
    if not reference_prompt:
        return None, "parse", f"{model_name} không tạo được reference_prompt hợp lệ."

    return reference_prompt, None, None


def _is_model_available(model_name):
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        names = {m.get("name", "") for m in models if isinstance(m, dict)}
        return model_name in names
    except requests.RequestException:
        # Khi không kiểm tra được tags, để request chính trả lỗi chi tiết từ Ollama.
        return True


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
    except Exception:
        return jsonify({"success": False, "models": []})


@app.route('/api/reference-analyze', methods=['POST'])
def reference_analyze():
    files = [f for f in request.files.getlist('images') if f and f.filename]
    if not files:
        return jsonify({"success": False, "error": "Vui lòng tải lên ít nhất một ảnh tham chiếu."})

    if len(files) > VISION_MAX_REFERENCE_IMAGES:
        return jsonify({
            "success": False,
            "error": f"Tối đa {VISION_MAX_REFERENCE_IMAGES} ảnh tham chiếu mỗi lần phân tích."
        })

    base64_images = _save_and_encode_uploads(files)
    if not base64_images:
        return jsonify({"success": False, "error": "Không đọc được dữ liệu ảnh tham chiếu."})

    system_prompt = (
        "Bạn là chuyên gia phân tích ảnh để tạo prompt text-to-image. "
        "Phân tích toàn bộ ảnh người dùng gửi (nhiều ảnh thì tổng hợp điểm chung và điểm nổi bật), "
        "sau đó TRẢ VỀ DUY NHẤT một JSON object hợp lệ, không có markdown, không có giải thích thêm. "
        "Schema bắt buộc: "
        "{\"subject\":\"...\",\"style\":\"...\",\"composition\":\"...\","
        "\"lighting\":\"...\",\"color_palette\":\"...\",\"camera\":\"...\","
        "\"mood\":\"...\",\"negative\":\"...\"}"
    )

    if not _is_model_available(VISION_PRIMARY_MODEL):
        return jsonify({
            "success": False,
            "error": f"Model phân tích ảnh {VISION_PRIMARY_MODEL} chưa có trong Ollama local (model not found).",
        })

    primary_timeout_seconds = _compute_primary_vision_timeout(len(base64_images))

    reference_prompt, reason, error_message = _run_vision_analysis(
        VISION_PRIMARY_MODEL,
        base64_images,
        primary_timeout_seconds,
        system_prompt,
    )

    if reference_prompt:
        return jsonify({
            "success": True,
            "reference_prompt": reference_prompt,
            "source_model": VISION_PRIMARY_MODEL,
        })

    if reason in {"request", "timeout", "model_error", "parse"}:
        return jsonify({
            "success": False,
            "error": (
                "Primary vision fail-fast (single-model semantic reference). "
                f"{error_message}"
            ),
        })

    return jsonify({"success": False, "error": error_message})


@app.route('/api/generate', methods=['POST'])
def generate():
    prompt = request.form.get('prompt', '').strip()
    reference_prompt = request.form.get('reference_prompt', '').strip()
    model = request.form.get('model', '')
    files = request.files.getlist('images')
    has_reference_images = any(file.filename for file in files)

    if has_reference_images and not reference_prompt:
        return jsonify({
            "success": False,
            "error": "Bạn đã tải ảnh tham chiếu nhưng chưa phân tích. Vui lòng bấm 'Phân tích ảnh tham chiếu' trước khi tạo ảnh."
        })

    final_prompt = prompt
    if reference_prompt:
        final_prompt = (
            f"{prompt}\n\n"
            f"Reference guidance (from user reference images):\n{reference_prompt}"
        ).strip()

    if not final_prompt:
        return jsonify({"success": False, "error": "Prompt không được để trống."})

    payload = {"model": model, "prompt": final_prompt, "stream": False}

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

    if isinstance(res_json, dict) and res_json.get('error'):
        return jsonify({"success": False, "error": f"Ollama báo lỗi: {res_json['error']}"})

    b64_raw = _extract_base64_candidate(res_json)
    b64_str = _normalize_base64_string(b64_raw)
    if not b64_str:
        keys = list(res_json.keys()) if isinstance(res_json, dict) else []
        return jsonify({
            "success": False,
            "error": f"Ollama không trả dữ liệu ảnh hợp lệ. Keys nhận được: {keys}",
        })

    try:
        image_bytes = base64.b64decode(b64_str, validate=True)
    except (binascii.Error, ValueError):
        return jsonify({"success": False, "error": "Dữ liệu ảnh base64 không hợp lệ từ Ollama."})

    if not image_bytes:
        return jsonify({"success": False, "error": "Dữ liệu ảnh rỗng từ Ollama."})

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


@app.route('/media/<folder>/<filename>')
def serve_media(folder, filename):
    target_dir = UPLOAD_DIR if folder == 'uploads' else OUTPUT_DIR
    return send_from_directory(target_dir, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
