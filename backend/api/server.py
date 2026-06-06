"""
Flask API：前端发图，后端返回预测 + 每层激活

POST /predict
    Body JSON: { "image": [784 floats]  // 28x28，值在 [0, 1] (1=黑，0=白)
                 或 "image": [1024 floats] // 32x32 }
    Response:  { "prediction": int,
                 "probs": [10 floats],
                 "activations": { layer_name: [...], ... } }

GET /health
    检查模型是否加载好
"""

import os
import sys
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 把项目根加入 path，方便 `python -m backend.api.server` 启动
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.model import LeNet5  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lenet5.npz")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

model = LeNet5(use_sparse_c3=True)
_weights_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        model.load(MODEL_PATH)
        _weights_loaded = True
        print(f"[server] loaded weights from {MODEL_PATH}")
    except Exception as e:
        print(f"[server] WARNING: failed to load weights ({e}), using random init")
else:
    print(f"[server] WARNING: no weights at {MODEL_PATH}, predictions will be random.")
    print(f"[server] Run `python -m backend.train` first.")


# ─────────────────────────────────────────────────────────────────────────────
# 图片预处理：前端 canvas -> 模型输入
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(image_array):
    """
    image_array: 28x28 或 32x32 的 list/array，值在 [0, 1]，1=有笔迹 (黑)。
    返回: (1, 1, 28, 28) float32，归一化到 [-1, 1] (1=有笔迹 → +1)。
    
    MNIST 训练时的约定是: 像素值高 = 数字笔迹。
    """
    arr = np.array(image_array, dtype=np.float32)
    n = arr.size
    if n == 28 * 28:
        arr = arr.reshape(28, 28)
    elif n == 32 * 32:
        arr = arr.reshape(32, 32)[2:30, 2:30]
    else:
        side = int(np.sqrt(n))
        arr = arr.reshape(side, side)
        # 如果不是 28 / 32，用简单的最近邻缩放
        if side != 28:
            from scipy.ndimage import zoom
            arr = zoom(arr, 28 / side, order=1)

    # 限制到 [0, 1]
    arr = np.clip(arr, 0.0, 1.0)
    # MNIST 是 [-1, 1]，且笔迹 (255 / 1.0) 是正
    arr = arr * 2.0 - 1.0
    return arr[None, None, :, :]   # (1, 1, 28, 28)


def serialize_activations(acts):
    """把 numpy 数组转成可 JSON 序列化的嵌套列表，并控制精度."""
    out = {}
    for name, arr in acts.items():
        # 去掉 batch 维 (这里 N=1)
        a = arr[0]
        # 控制精度 4 位小数，节省传输大小
        out[name] = np.round(a.astype(np.float32), 4).tolist()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "weights_loaded": _weights_loaded,
        "model": "LeNet-5",
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "missing 'image' field"}), 400
    try:
        x = preprocess(data["image"])
    except Exception as e:
        return jsonify({"error": f"preprocess failed: {e}"}), 400

    logits, acts = model.forward(x, collect_activations=True)
    probs = acts["probs"][0]
    prediction = int(np.argmax(probs))

    return jsonify({
        "prediction": prediction,
        "probs": probs.astype(float).tolist(),
        "activations": serialize_activations(acts),
        "weights_loaded": _weights_loaded,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 静态文件：把 frontend/ 直接 serve 出去，避免跨域 + 额外起服务器
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
