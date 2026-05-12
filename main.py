from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import os

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
CORS(app)  # ✅ Allow browser requests (required for web frontend)

# -----------------------
# Tesseract path
# Commented out for Railway (Linux); only needed on Windows locally
# -----------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -----------------------
# Load YOLO models
# -----------------------
object_model   = YOLO("yolov8n.pt")
currency_model = YOLO("best.pt")

labels = [
    "1000_back", "1000_front",
    "100_back",  "100_front",
    "10_back",   "10_front",
    "20_back",   "20_front",
    "5000_back", "5000_front",
    "500_back",  "500_front",
    "50_back",   "50_front"
]

# -----------------------
# Home route
# -----------------------
@app.route("/")
def home():
    return {"message": "Vision Flask API is running"}

# -----------------------
# Object Detection (YOLOv8)
# POST /detect  —  field: "file"
# Returns: { objects: [...], message: "..." }
# -----------------------
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file  = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = object_model(img, conf=0.25)
    objects = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            objects.append(object_model.names[cls_id])
    objects = list(set(objects))

    return jsonify({
        "objects": objects,
        "message": "Detected: " + ", ".join(objects) if objects else "No objects detected"
    })

# -----------------------
# Text Detection (Tesseract OCR)
# POST /detect-text  —  field: "file"
# Returns: { text: "...", message: "..." }
# -----------------------
@app.route("/detect-text", methods=["POST"])
def detect_text():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file  = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    return jsonify({
        "text":    text.strip(),
        "message": "Detected text" if text.strip() else "No text detected"
    })

# -----------------------
# Currency Detection (YOLOv8 classify)
# POST /detect-currency  —  field: "file"
# Returns: { currency: [{label, confidence}], message: "..." }
# -----------------------
@app.route("/detect-currency", methods=["POST"])
def detect_currency():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file  = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results  = currency_model(img)
    detected = []
    for r in results:
        if r.probs is not None:
            top1       = r.probs.top1
            label      = r.names[top1]
            confidence = float(r.probs.top1conf)
            detected.append({
                "label":      label,
                "confidence": round(confidence, 3)
            })

    return jsonify({
        "currency": detected,
        "message":  f"Detected: {detected}" if detected else "No currency detected"
    })

# -----------------------
# Entry point
# Uses Railway's PORT env var, falls back to 5000 locally
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
