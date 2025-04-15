from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os
import shutil
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://cancer-detection-nodejs.onrender.com"]}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = YOLO("EX3.pt")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clear_directories():
    for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")

def resize_image(input_path, max_size=640):
    try:
        img = Image.open(input_path)
        img.thumbnail((max_size, max_size))
        resized_path = input_path.replace(".", "_resized.")
        img.save(resized_path)
        return resized_path
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return input_path

@app.route("/")
def index():
    return "✅ Python API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Received predict request")
    clear_directories()

    if 'image' not in request.files:
        logger.error("No image uploaded")
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    filename = image.filename
    temp_path = os.path.join(UPLOAD_DIR, filename)
    logger.info(f"Saving image to {temp_path}")
    image.save(temp_path)

    # Resize ภาพ
    temp_path = resize_image(temp_path)

    results = []
    for conf in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        predict_dir_name = f"conf_{int(conf * 100)}"
        predict_dir_path = os.path.join(OUTPUT_DIR, predict_dir_name)

        try:
            logger.info(f"Running prediction with conf={conf}")
            model.predict(
                source=temp_path,
                save=True,
                save_txt=False,
                conf=conf,
                project=OUTPUT_DIR,
                name=predict_dir_name
            )

            result_path = None
            if os.path.exists(predict_dir_path):
                for file in os.listdir(predict_dir_path):
                    if file.endswith((".jpg", ".png")):
                        ext = file.split(".")[-1]
                        new_filename = f"predicted_conf{int(conf * 100)}.{ext}"
                        new_file_path = os.path.join(OUTPUT_DIR, new_filename)
                        shutil.move(os.path.join(predict_dir_path, file), new_file_path)
                        result_path = f"/outputs/{new_filename}"
                        break
                shutil.rmtree(predict_dir_path, ignore_errors=True)

            if result_path:
                results.append({"confidence": conf, "path": result_path})
            else:
                results.append({"confidence": conf, "error": "No result image generated"})

        except Exception as e:
            logger.error(f"Prediction failed for conf={conf}: {e}")
            results.append({"confidence": conf, "error": str(e)})

    if os.path.exists(temp_path):
        os.remove(temp_path)

    logger.info(f"Prediction completed for {filename}, results: {results}")
    if any(r.get("path") for r in results):
        return jsonify({
            "message": "Prediction completed",
            "results": results
        })
    else:
        return jsonify({"error": "No results generated", "results": results}), 500

@app.route("/outputs/<filename>")
def serve_output(filename):
    response = send_from_directory(OUTPUT_DIR, filename)
    response.headers["Access-Control-Allow-Origin"] = "https://cancer-detection-nodejs.onrender.com"
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)