from flask import Flask, request, jsonify
import os
import subprocess
from datetime import datetime

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    filename = f"input_{datetime.now().timestamp()}.jpg"
    image_path = os.path.join("static", filename)
    image.save(image_path)

    # เรียก YOLO script
    command = f"python yolov8_predict.py --source {image_path}"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500
        return jsonify({"message": "Prediction complete", "image": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

app.run(host="0.0.0.0", port=8000)
