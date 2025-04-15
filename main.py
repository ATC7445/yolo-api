from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os
import shutil

app = Flask(__name__)
model = YOLO("EX3.pt")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/")
def index():
    return "✅ Python API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    filename = image.filename
    temp_path = os.path.join(UPLOAD_DIR, filename)
    image.save(temp_path)

    # ใช้ confidence เดียวเพื่อความเร็ว
    conf = 0.6
    predict_dir_name = f"conf_{int(conf * 100)}"
    predict_dir_path = os.path.join(OUTPUT_DIR, predict_dir_name)

    # รัน prediction
    model.predict(
        source=temp_path,
        save=True,
        save_txt=False,
        conf=conf,
        project=OUTPUT_DIR,
        name=predict_dir_name
    )

    # หาไฟล์ผลลัพธ์ และย้ายขึ้น root /outputs/
    result_path = None
    if os.path.exists(predict_dir_path):
        for file in os.listdir(predict_dir_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                ext = file.split(".")[-1]
                new_filename = f"predicted_conf60.{ext}"
                new_file_path = os.path.join(OUTPUT_DIR, new_filename)
                shutil.move(os.path.join(predict_dir_path, file), new_file_path)
                result_path = f"/outputs/{new_filename}"
                break
        shutil.rmtree(predict_dir_path)

    if result_path:
        return jsonify({
            "message": "Prediction completed",
            "path": result_path
        })
    else:
        return jsonify({"error": "No result image generated"}), 500

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
