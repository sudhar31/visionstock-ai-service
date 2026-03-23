from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# Load model once
model = YOLO("yolov8n.pt")  # YOLO will auto-download if not present

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "VisionStock AI Service Running 🚀"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    # Save image temporarily
    filename = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # Run YOLO
    results = model(filepath)

    #Generate Annotated Image (Phase 6)
    annotated_filename = "result_" + filename
    annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
    results[0].save(filename=annotated_path)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0])
            })

    # Delete temp file
    #os.remove(filepath)

    return jsonify({
        "total_objects": len(detections),
        "detections": detections,
        "annotated_image" : annotated_filename
    })

@app.route("/result/<filename>")
def get_result_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)