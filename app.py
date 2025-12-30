from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import numpy as np
import os

from keras_facenet import FaceNet
from db.firestore import (
    student_exists,
    save_student,
    get_all_students,
    mark_attendance,
    get_attendance
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)

# MediaPipe Face Detector
options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(
        model_asset_path="face_detector.tflite"
    ),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)

face_detector = vision.FaceDetector.create_from_options(options)

# Face Embedding Model
embedder = FaceNet()

# -----------------------------
# Webcam

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -----------------------------
# Globals
# -----------------------------
registered_students = []
attendance_marked = set()

status_message = ""
SIMILARITY_THRESHOLD = 0.6

# Register state
register_start_time = None
registration_done = False
registration_done_time = None
last_embed_time = 0   # throttle embeddings

# -----------------------------
# Helpers
# -----------------------------
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape[0] != 512 or b.shape[0] != 512:
        return -1.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# UI ROUTES
# -----------------------------
@app.route("/")
def main():
    return render_template("index.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/attendance")
def attendance_page():
    records = get_attendance()
    return render_template("attendance.html", records=records)


# -----------------------------
# VIDEO FEED
# -----------------------------
@app.route("/video_feed")
def video_feed():
    mode = request.args.get("mode", "idle")
    student_id = request.args.get("id")
    student_name = request.args.get("name")

    return Response(
        generate_frames(mode, student_id, student_name),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# -----------------------------
# FRAME GENERATOR
# -----------------------------
def generate_frames(mode, student_id=None, student_name=None):
    global status_message
    global registered_students, attendance_marked
    global register_start_time, registration_done, registration_done_time
    global last_embed_time

    if mode in ("detect", "attendance") and not registered_students:
        registered_students = get_all_students()
        attendance_marked.clear()
        print(f"Loaded {len(registered_students)} students")

    if mode == "register" and register_start_time is None and not registration_done:
        register_start_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_count = 0

        try:
            detection_result = face_detector.detect(mp_image)

            if detection_result and detection_result.detections:
                for detection in detection_result.detections:
                    face_count += 1
                    bbox = detection.bounding_box

                    x1 = max(0, bbox.origin_x)
                    y1 = max(0, bbox.origin_y)
                    x2 = min(frame.shape[1], x1 + bbox.width)
                    y2 = min(frame.shape[0], y1 + bbox.height)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    # ---------- REGISTER ----------
                    if mode == "register" and student_id and student_name:
                        if not registration_done and register_start_time is not None:
                            elapsed = int(time.time() - register_start_time)
                            status_message = f"Registering... ({elapsed}s)"

                            if elapsed >= 2:
                                face = cv2.resize(face, (160, 160))
                                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                face = np.expand_dims(face, axis=0)

                                if not student_exists(student_id):
                                    emb = embedder.embeddings(face)[0]
                                    save_student(student_id, student_name, emb.tolist())
                                    status_message = "Registered"
                                else:
                                    status_message = "Already Registered"

                                registration_done = True
                                registration_done_time = time.time()
                                register_start_time = None

                    # ---------- DETECT / ATTENDANCE ----------
                    if mode in ("detect", "attendance") and registered_students:
                        now = time.time()
                        if now - last_embed_time < 0.4:
                            continue  # throttle heavy inference
                        last_embed_time = now

                        face = cv2.resize(face, (160, 160))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face = np.expand_dims(face, axis=0)

                        live_emb = embedder.embeddings(face)[0]

                        best_match = None
                        best_score = -1.0

                        for student in registered_students:
                            score = cosine_similarity(live_emb, student["embedding"])
                            if score > best_score:
                                best_score = score
                                best_match = student

                        if best_match and best_score > SIMILARITY_THRESHOLD:
                            sid = best_match["id"]

                            if mode == "attendance":
                                if sid not in attendance_marked:
                                    mark_attendance(
                                        sid,
                                        best_match["name"],
                                        float(best_score)
                                    )
                                    attendance_marked.add(sid)
                                status_message = "Attendance Posted"
                            else:
                                status_message = ""

                            cv2.putText(
                                frame,
                                sid,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0) if mode == "attendance" else (255, 255, 0),
                                2
                            )
                        else:
                            status_message = (
                                "Attendance Not Posted"
                                if mode == "attendance"
                                else "Not Detected"
                            )

            # ---------- CLEAR REGISTER MESSAGE ----------
            if registration_done and registration_done_time:
                if time.time() - registration_done_time > 2:
                    status_message = ""
                    registration_done = False
                    registration_done_time = None

        except Exception as e:
            print("ERROR:", e)
            status_message = "Processing error"

        cv2.putText(
            frame,
            f"{mode.upper()} | Faces: {face_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        if status_message:
            cv2.putText(
                frame,
                status_message,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255) if "Not" not in status_message else (0, 0, 255),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False, threaded=True)
