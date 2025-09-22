import cv2
import mediapipe as mp
import numpy as np
import tensorflow.lite as tflite
import json
import os
import time
import random
 
interpreter = tflite.Interpreter(model_path=r"D:/6th/vsc/week3/facenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
embedding_size = output_details[0]['shape'][1]

def safe_cosine(u, v, eps=1e-6):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < eps or norm_v < eps:
        return float("inf")
    cosine_sim = np.dot(u, v) / (norm_u * norm_v)
    return 1 - cosine_sim  #distance: semakin kecil semakin mirip

EMBEDDINGS_FILE = "D:/6th/vsc/week3/face_embeddings3.json"

def load_embeddings_json(filename=EMBEDDINGS_FILE):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data = json.load(f)
                names = []
                embeddings_list = []
                for name, emb_list in data.items():
                    for emb in emb_list:
                        if len(emb) == embedding_size:
                            names.append(name)
                            embeddings_list.append(emb)
                embeddings = np.array(embeddings_list, dtype=np.float32)
                return names, embeddings
            except json.JSONDecodeError:
                print("Error: Corrupted JSON file. Resetting database.")
                return [], np.empty((0, embedding_size), dtype=np.float32)
    return [], np.empty((0, embedding_size), dtype=np.float32)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Variabel Liveness Detection
last_blink_time = time.time()
stable_frame_count = 0
prev_nose = None
BLINK_TIME_THRESHOLD = 4.0
NOSE_MOVEMENT_THRESHOLD = 2.0
STABLE_FRAME_COUNT_THRESHOLD = 30
EAR_THRESHOLD = 0.2
COLOR_DIFF_THRESHOLD = 30

# Indeks landmark mata
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]

# Menghitung EAR (Eye Aspect Ratio) 
def compute_ear(landmarks, eye_indices, img_width, img_height):
    points = [(landmarks[idx].x * img_width, landmarks[idx].y * img_height) for idx in eye_indices]
    p1, p2, p3, p4, p5, p6 = points
    dist1 = np.linalg.norm(np.array(p2) - np.array(p6))
    dist2 = np.linalg.norm(np.array(p3) - np.array(p5))
    dist3 = np.linalg.norm(np.array(p1) - np.array(p4))
    return (dist1 + dist2) / (2.0 * dist3)

# Deteksi spoofing warna
def detect_color_spoof(face_crop):
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    return abs(h_mean - s_mean) < COLOR_DIFF_THRESHOLD

COSINE_THRESHOLD = 0.4  #Threshold Face Recognition

# Variabel Gesture Detection (dengan Head Pose Estimation)
gestures = ["Turn Left", "Turn Right"]
current_gesture = random.choice(gestures)
gesture_count = 0
required_gestures = 5
yaw_threshold = 15  #threshold derajat untuk mendeteksi gerakan ke kiri/kanan
gesture_delay = 2   #Minimum seconds between valid gesture detectionsd
last_gesture_time = 0   

# Variabel untuk smoothing yaw
smoothed_yaw = None
yaw_alpha = 0.2  #faktor smoothing, nilai lebih rendah => lebih banyak smoothing
yaw_dead_zone = 10  #Additional dead zone to prevent minor movements from triggering 

# 3D model points dari landmark wajah (estimasi kasar, satuan mm)
# Menambahkan 6 titik: Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
model_points = np.array([
    (0.0, 0.0, 0.0),         # Nose tip (landmark index 1)
    (0.0, -63.6, -12.5),      # Chin (landmark index 152)
    (-42.0, 32.0, -26.0),     # Left eye left corner (landmark index 33)
    (42.0, 32.0, -26.0),      # Right eye right corner (landmark index 263)
    (-28.0, -20.0, -30.0),    # Left mouth corner (landmark index 61)
    (28.0, -20.0, -30.0)      # Right mouth corner (landmark index 291)
], dtype="double")

cap = cv2.VideoCapture(0) # Mulai Kamera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(rgb_frame)
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

            mesh_results = face_mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

            # Gesture Detection 
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                h_crop, w_crop, _ = face_crop.shape

                # Ambil 2D image points dari 6 landmark yang sesuai
                image_points = np.array([
                    (landmarks[1].x * w_crop, landmarks[1].y * h_crop),     # Nose tip
                    (landmarks[152].x * w_crop, landmarks[152].y * h_crop),   # Chin
                    (landmarks[33].x * w_crop, landmarks[33].y * h_crop),     # Left eye corner
                    (landmarks[263].x * w_crop, landmarks[263].y * h_crop),   # Right eye corner
                    (landmarks[61].x * w_crop, landmarks[61].y * h_crop),     # Left mouth corner
                    (landmarks[291].x * w_crop, landmarks[291].y * h_crop)    # Right mouth corner
                ], dtype="double")

                focal_length = w_crop  
                center = (w_crop / 2, h_crop / 2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double"
                )
                dist_coeffs = np.zeros((4, 1))  

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
                    singular = sy < 1e-6
                    if not singular:
                        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                    else:
                        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
                        z_angle = 0
                    x_angle = np.degrees(x_angle)
                    y_angle = np.degrees(y_angle)
                    z_angle = np.degrees(z_angle)

                    # Smoothing yaw angle
                    if smoothed_yaw is None:
                        smoothed_yaw = y_angle
                    else:
                        smoothed_yaw = yaw_alpha * y_angle + (1 - yaw_alpha) * smoothed_yaw

                    # Tampilkan nilai yaw untuk debugging
                    cv2.putText(frame, f"Yaw: {smoothed_yaw:.1f}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.putText(frame, f"Please {current_gesture}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                    # Only process gesture detection if sufficient time has passed
                    if current_time - last_gesture_time >= gesture_delay:
                        if current_gesture == "Turn Left" and smoothed_yaw < -(yaw_threshold + yaw_dead_zone):
                            gesture_count += 1
                            print(f"Gesture '{current_gesture}' detected! Yaw: {smoothed_yaw:.1f}")
                            current_gesture = random.choice(gestures)
                            last_gesture_time = current_time  # Reset timer

                        elif current_gesture == "Turn Right" and smoothed_yaw > (yaw_threshold + yaw_dead_zone):
                            gesture_count += 1
                            print(f"Gesture '{current_gesture}' detected! Yaw: {smoothed_yaw:.1f}")
                            current_gesture = random.choice(gestures)
                            last_gesture_time = current_time  # Reset timer


                        # Jika gesture yang diinginkan telah tercapai, keluar dari program
                        if gesture_count >= required_gestures:
                            cv2.putText(frame, "All gestures completed.", (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            cv2.imshow('Face Recognition & Liveness', frame)
                            print("Required gestures completed. Exiting...")
                            time.sleep(2)
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()

            # Liveness & Face Recognition Processing
            liveness_flag = True

            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0].landmark
                h_crop, w_crop, _ = face_crop.shape

                left_ear = compute_ear(landmarks, left_eye_indices, w_crop, h_crop)
                right_ear = compute_ear(landmarks, right_eye_indices, w_crop, h_crop)
                ear = (left_ear + right_ear) / 2
                if ear < EAR_THRESHOLD: 
                    last_blink_time = current_time 

                nose = landmarks[1]
                nose_coord = (nose.x * w_crop, nose.y * h_crop)
                if prev_nose is not None:
                    movement = np.linalg.norm(np.array(nose_coord) - np.array(prev_nose))
                    stable_frame_count = stable_frame_count + 1 if movement < NOSE_MOVEMENT_THRESHOLD else 0
                prev_nose = nose_coord

                if detect_color_spoof(face_crop):
                    liveness_flag = False

                if (current_time - last_blink_time) > BLINK_TIME_THRESHOLD and stable_frame_count > STABLE_FRAME_COUNT_THRESHOLD:
                    liveness_flag = False
            else:
                liveness_flag = False

            if liveness_flag:
                face_resized = cv2.resize(face_crop, (160, 160))
                face_normalized = face_resized.astype(np.float32) / 127.5 - 1
                face_input = np.expand_dims(face_normalized, axis=0)

                interpreter.set_tensor(input_details[0]['index'], face_input)
                interpreter.invoke()
                face_embedding = interpreter.get_tensor(output_details[0]['index'])[0].flatten()

                names, embeddings = load_embeddings_json()
                if len(embeddings) > 0:
                    for i, db_embedding in enumerate(embeddings):
                        dist = safe_cosine(face_embedding, db_embedding.flatten())
                        if dist < min_dist:
                            min_dist = dist
                            identity = names[i] if dist < COSINE_THRESHOLD else "Unknown"

            label = identity if liveness_flag else "Spoof Detected"
            color = (0, 255, 0) if liveness_flag else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Face Recognition & Liveness', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()