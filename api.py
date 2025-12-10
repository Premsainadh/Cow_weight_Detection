import os
import io
import math
import requests
import joblib
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# === Load ML Scale Model ===
scale_model = joblib.load("scale_model.pkl")

# Global flag
flag = 0


# === Compress Image In-Memory ===
def compress_image_in_memory(image_path, max_size_mb=2):
    global flag
    flag = 1

    max_bytes = max_size_mb * 1024 * 1024
    original_size = os.path.getsize(image_path)

    if original_size <= max_bytes:
        flag = 0
        return open(image_path, "rb")

    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    quality = 95
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)

    while buffer.getbuffer().nbytes > max_bytes and quality > 10:
        quality -= 5
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)

    buffer.seek(0)
    return buffer


# === Compute Measurements Using Your ML Model ===
def compute_cow_measurements(pred, scale_factor=None):
    global flag

    if flag == 1:
        kps = {kp["class"]: (kp["x"]/2, kp["y"]/2) for kp in pred["keypoints"]}
    else:
        kps = {kp["class"]: (kp["x"], kp["y"]) for kp in pred["keypoints"]}
    flag = 0

    def euclidean(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    pixel_body_length = euclidean(kps["Shoulder1"], kps["Tail-base"])
    pixel_heart_dia = euclidean(kps["Shoulder2"], kps["withers"])

    bbox_w = pred["width"]
    bbox_h = pred["height"]

    feature_vector = np.array([[
        pixel_body_length,
        bbox_w,
        bbox_h,
        kps["Shoulder1"][0], kps["Shoulder1"][1],
        kps["Tail-base"][0], kps["Tail-base"][1],
        kps["withers"][0],   kps["withers"][1],
        kps["Shoulder2"][0], kps["Shoulder2"][1]
    ]])

    scale_factor = scale_model.predict(feature_vector)[0]

    real_body_length = pixel_body_length * scale_factor
    real_heart_dia = pixel_heart_dia * scale_factor

    def ramanujan_girth(D1, D2):
        a = D1 / 2.0
        b = D2 / 2.0
        term = 3*(a + b) - ((3*a + b)*(a + 3*b))**0.5
        return math.pi * term

    heart_girth = ramanujan_girth(real_heart_dia, 1.08 * real_heart_dia)

    weight = (real_body_length * (heart_girth ** 2)) / 660

    return {
        "pixel_body_length": pixel_body_length,
        "scale_factor": scale_factor,
        "body_length_in": real_body_length,
        "heart_girth_in": heart_girth,
        "weight": weight
    }


# === Run Roboflow Keypoint Detection ===
def run_inference(image_buffer, api_key, model_id):
    img = Image.open(image_buffer).convert("RGB")
    image_buffer.seek(0)

    response = requests.post(
        f"https://detect.roboflow.com/{model_id}?api_key={api_key}&format=json",
        files={"file": ("image.jpg", image_buffer, "image/jpeg")}
    )

    result = response.json()

    if "predictions" not in result or len(result["predictions"]) == 0:
        return {"error": "No cow detected in image!"}

    pred = result["predictions"][0]
    return compute_cow_measurements(pred)


# === Flask Routes ===
@app.route("/")
def home():
    return jsonify({"message": "Cattle Weight Estimation API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded!"}), 400

    uploaded_image = request.files["image"]
    temp_path = "temp_image.jpg"
    uploaded_image.save(temp_path)

    compressed_image = compress_image_in_memory(temp_path)
    os.remove(temp_path)

    api_key = "wUHnmbcsjGFnk0qkplqR"
    model_id ="cattle-body-measurements-wtegw/1"

    if not api_key or not model_id:
        return jsonify({"error": "API key or Model ID missing"}), 500

    result = run_inference(compressed_image, api_key, model_id)
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
