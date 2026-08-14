"""
CSCAN Brain Tumor MRI Classification - Flask Backend
======================================================

Loads the existing, already-trained CSCAN model (ConvNeXt-Tiny + Swin-T dual
branch, Adaptive Weighted Fusion, Cross-Attention Fusion, DCRF refinement,
GeM pooling classifier head) and serves predictions over a small web UI.

Nothing about the model architecture, training, or class ordering is changed
here. This file only wires the existing `models/cscan.py` implementation and
the existing `utils/transforms.py` / `utils/config.py` pipeline into a Flask
app + SQLite history log.
"""

import os
import sqlite3
import traceback
import uuid
from datetime import datetime

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from models.cscan import CSCAN
from utils.config import CSCANConfig
from utils.transforms import get_val_test_transforms

# -----------------------------------------------------------------------
# Paths & basic Flask configuration
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "database.db")

# The trained checkpoint lives at the project root, per the required
# project structure (CSCAN_WebApp/best_cscan_model.pth).
MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_cscan_model.pth")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# -----------------------------------------------------------------------
# Class ordering — MUST match utils/dataset.py / utils/config.py exactly.
# CSCANConfig.CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
# The index a class occupies here is the same index the model was trained
# to output, so this ordering is never hand-edited.
# -----------------------------------------------------------------------
CLASS_NAMES = CSCANConfig.CLASSES
CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor",
}

# -----------------------------------------------------------------------
# Device selection (reuses the same logic already defined in
# utils/config.py: CUDA if available, else CPU)
# -----------------------------------------------------------------------
DEVICE = CSCANConfig.DEVICE

# -----------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------
def init_db():
    """Create the predictions table if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(filename, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (filename, prediction, confidence, created_at) "
        "VALUES (?, ?, ?, ?)",
        (filename, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def fetch_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, filename, prediction, confidence, created_at "
        "FROM predictions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


# -----------------------------------------------------------------------
# Model loading (once, at process startup)
# -----------------------------------------------------------------------
model = None
model_load_error = None
val_test_transform = get_val_test_transforms(CSCANConfig.IMAGE_SIZE)


def load_model():
    """
    Builds the CSCAN architecture exactly as defined in models/cscan.py
    (using the same hyperparameters the model was trained with, from
    utils/config.py), then loads the trained checkpoint weights on top.

    Any failure here (missing checkpoint, corrupt file, architecture/
    checkpoint mismatch) is captured so the Flask app can still start and
    report a clear error to the user instead of crashing.
    """
    global model, model_load_error
    try:
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found at '{MODEL_CHECKPOINT_PATH}'. "
                f"Place your trained 'best_cscan_model.pth' file in the "
                f"project root (next to app.py)."
            )

        net = CSCAN(
            num_classes=CSCANConfig.NUM_CLASSES,
            img_size=CSCANConfig.IMAGE_SIZE,
            dropout_rate=CSCANConfig.DROPOUT_RATE,
            pretrained=CSCANConfig.PRETRAINED,
            freeze_stages=CSCANConfig.FREEZE_STAGES,
            use_dcrf=CSCANConfig.USE_DCRF,
        )

        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
        # Support both a raw state_dict (what train.py/test.py save) and a
        # checkpoint dict that wraps it under a common key, just in case.
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        net.load_state_dict(state_dict)
        net.to(DEVICE)
        net.eval()

        model = net
        model_load_error = None
        print(f"[CSCAN] Model loaded successfully on device: {DEVICE}")
    except Exception as exc:  # noqa: BLE001 - we want to surface any load error
        model = None
        model_load_error = str(exc)
        print(f"[CSCAN] ERROR loading model: {model_load_error}")


# -----------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def run_inference(image_path):
    """
    Applies the exact validation/test preprocessing pipeline
    (utils/transforms.get_val_test_transforms: CLAHE -> Resize -> ToTensor
    -> Normalize) and runs a forward pass through the trained CSCAN model.

    If CSCANConfig.USE_TTA is enabled (as it is in test.py), predictions
    are averaged over the image and its horizontal flip - the same
    test-time augmentation used to produce the reported test metrics.

    Returns (predicted_class_key, confidence_float_0_to_100).
    """
    with open(image_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

    tensor = val_test_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)

        if getattr(CSCANConfig, "USE_TTA", False):
            flipped_logits = model(torch.flip(tensor, dims=[3]))
            probs = 0.5 * (F.softmax(logits, dim=1) + F.softmax(flipped_logits, dim=1))
        else:
            probs = F.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence_pct = confidence.item() * 100.0
    return predicted_class, confidence_pct


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_ready=(model is not None), model_error=model_load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": (
                "Model is not loaded. "
                + (model_load_error or "Unknown error loading checkpoint.")
            ),
        }), 503

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file was uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Please upload a JPG, JPEG, or PNG image.",
        }), 400

    # Build a safe, unique filename so concurrent uploads never collide.
    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(save_path)
    except Exception:
        return jsonify({"success": False, "error": "Failed to save the uploaded file."}), 500

    try:
        predicted_class, confidence_pct = run_inference(save_path)
    except UnidentifiedImageError:
        os.remove(save_path)
        return jsonify({
            "success": False,
            "error": "The uploaded file is not a valid image.",
        }), 400
    except Exception:
        traceback.print_exc()
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({
            "success": False,
            "error": "An error occurred while running the model on this image.",
        }), 500

    display_name = CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class)

    try:
        save_prediction(unique_name, display_name, round(confidence_pct, 2))
    except Exception:
        traceback.print_exc()
        # Prediction still succeeded even if logging to the DB failed.

    return jsonify({
        "success": True,
        "prediction": display_name,
        "confidence": round(confidence_pct, 2),
        "filename": unique_name,
        "image_url": url_for("static", filename=f"uploads/{unique_name}"),
    })


@app.route("/history")
def history():
    rows = fetch_history()
    return render_template("history.html", records=rows)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
