import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont  # Step 3: PIL for annotation
import cv2
import numpy as np

# Flask app setup
app = Flask(__name__, static_folder="../frontend/build", static_url_path="/")
CORS(app)

# Folders for uploads and results
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

PIXEL_SPACING = 0.5  # mm/pixel

@app.route("/upload-scan", methods=["POST"])
def upload_scan():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Read image using OpenCV (grayscale)
    img_cv = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        return jsonify({"error": "Failed to read image"}), 400

    # Thresholding to detect stones
    _, thresh = cv2.threshold(img_cv, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert OpenCV image to PIL image for annotation
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()

    stones = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        width_mm = w * PIXEL_SPACING
        height_mm = h * PIXEL_SPACING

        # Determine stone location
        vert = "top" if y + h/2 < img_cv.shape[0]/3 else "bottom" if y + h/2 > 2*img_cv.shape[0]/3 else "center"
        horiz = "left" if x + w/2 < img_cv.shape[1]/3 else "right" if x + w/2 > 2*img_cv.shape[1]/3 else "center"
        location = f"{vert}-{horiz}"

        stones.append({
            "width_mm": round(width_mm, 2),
            "height_mm": round(height_mm, 2),
            "location": location
        })

        # Step 3: Annotate using PIL
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.text((x, y - 15), location, fill="green", font=font)

    # Save annotated image
    annotated_path = os.path.join(RESULT_FOLDER, "annotated_" + filename)
    img_pil.save(annotated_path)

    response = {
        "annotated_image": f"http://127.0.0.1:5000/results/annotated_{filename}",
        "stones": stones,
        "report_text": f"{len(stones)} stone(s) detected."
    }
    return jsonify(response)

@app.route("/results/<filename>")
def serve_results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# Serve React frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    print("Starting Flask server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
