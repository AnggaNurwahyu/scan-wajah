from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow.lite as tflite
import mediapipe as mp
import json
import os

# ========== Inisialisasi Model ==========
interpreter = tflite.Interpreter(model_path="facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
embedding_size = output_details[0]['shape'][1]

# ========== Load Embeddings ==========
EMBEDDINGS_FILE = "face_embeddings3.json"

def safe_cosine(u, v, eps=1e-6):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < eps or norm_v < eps:
        return float("inf")
    cosine_sim = np.dot(u, v) / (norm_u * norm_v)
    return 1 - cosine_sim

def load_embeddings_json(filename=EMBEDDINGS_FILE):
    if not os.path.exists(filename):
        return [], np.empty((0, embedding_size), dtype=np.float32)
    with open(filename, "r") as f:
        data = json.load(f)
        names, embeddings_list = [], []
        for name, emb_list in data.items():
            for emb in emb_list:
                names.append(name)
                embeddings_list.append(emb)
        embeddings = np.array(embeddings_list, dtype=np.float32)
        return names, embeddings

# ========== MediaPipe ==========
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

COSINE_THRESHOLD = 0.4

# ========== Fungsi Proses ==========
def process_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(rgb_frame)

    names, embeddings = load_embeddings_json()
    identity = "Unknown"
    min_dist = float("inf")

    if detection_results.detections:
        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h_frame, w_frame, _ = frame.shape
            x, y = int(bboxC.xmin * w_frame), int(bboxC.ymin * h_frame)
            w_box, h_box = int(bboxC.width * w_frame), int(bboxC.height * h_frame)

            face_crop = frame[y:y+h_box, x:x+w_box]
            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (160, 160))
            face_normalized = face_resized.astype(np.float32) / 127.5 - 1
            face_input = np.expand_dims(face_normalized, axis=0)

            interpreter.set_tensor(input_details[0]['index'], face_input)
            interpreter.invoke()
            face_embedding = interpreter.get_tensor(output_details[0]['index'])[0].flatten()

            for i, db_embedding in enumerate(embeddings):
                if len(db_embedding) == embedding_size:
                    dist = safe_cosine(face_embedding, db_embedding)
                    if dist < min_dist:
                        min_dist = dist
                        identity = names[i] if dist < COSINE_THRESHOLD else "Unknown"

    return {
        "identity": identity,
        "valid": identity != "Unknown",
        "distance": float(min_dist)
    }

# ========== Flask API ==========
app = Flask(__name__)

@app.route("/scan", methods=["POST"])
def scan():
    data = request.json
    img_base64 = data.get("image")

    if img_base64 is None:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 → numpy
    img_bytes = base64.b64decode(img_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = process_image(frame)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
