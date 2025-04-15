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

    # สร้าง predicted_conf40-90
    for i in range(4, 10):
        conf = i / 10.0
        model.predict(
            source=temp_path,
            save=True,
            save_txt=False,
            conf=conf,
            project=OUTPUT_DIR,
            name=f"conf_{int(conf * 100)}"
        )

    # รวมไฟล์ไปไว้ใน root outputs/
    result_paths = []
    for folder in os.listdir(OUTPUT_DIR):
        folder_path = os.path.join(OUTPUT_DIR, folder)
        if os.path.isdir(folder_path) and folder.startswith("conf_"):
            conf_level = folder.split("_")[1]
            for file in os.listdir(folder_path):
                if file.endswith(".jpg") or file.endswith(".png"):
                    new_filename = f"predicted_conf{conf_level}.{file.split('.')[-1]}"
                    new_file_path = os.path.join(OUTPUT_DIR, new_filename)
                    shutil.move(os.path.join(folder_path, file), new_file_path)
                    result_paths.append(f"/outputs/{new_filename}")
            shutil.rmtree(folder_path)

    return jsonify({
        "message": "Prediction completed",
        "results": result_paths
    })

@app.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
