from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import defaultdict
from datetime import datetime
from google import genai
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

# -----------------------------
# Gemini Client
# -----------------------------
genai_client = None
if os.getenv("GOOGLE_API_KEY"):
    try:
        genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        print("Gemini client initialized")
    except Exception as e:
        print("Gemini init failed:", e)

# -----------------------------
# MediaPipe Face Detector
# -----------------------------
options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(
        model_asset_path="face_detector.tflite"
    ),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5
)

face_detector = vision.FaceDetector.create_from_options(options)

# -----------------------------
# Face Embedding Model
# -----------------------------
embedder = FaceNet()

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)


# -----------------------------
# Globals
# -----------------------------
registered_students = []
attendance_marked = set()

status_message = ""
SIMILARITY_THRESHOLD = 0.6

register_start_time = None
registration_done = False
registration_done_time = None
last_embed_time = 0

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
# ROUTES
# -----------------------------
@app.route("/")
def main():
    return render_template("index.html")

@app.route("/attendance")
def attendance_page():
    records = get_attendance()
    return render_template("attendance.html", records=records)

@app.route("/analytics")
def analytics():
    records = get_attendance()

    if not records:
        return render_template(
            "analytics.html",
            stats={
                "total_records": 0,
                "unique_students": 0,
                "today_count": 0,
                "avg_confidence": 0
            },
            trend_data={},
            insights={
                "overview": "No attendance data available yet.",
                "trend": "Attendance trends will appear once records are created.",
                "action": "Start marking attendance to unlock analytics."
            }
        )

    unique_students = set()
    today = datetime.utcnow().date()
    today_present = set()

    for r in records:
        unique_students.add(r["student_id"])
        if r["timestamp"].date() == today:
            today_present.add(r["student_id"])

    stats = {
        "total_records": len(records),
        "unique_students": len(unique_students),
        "today_count": len(today_present),
        "avg_confidence": round(
            sum(r["confidence"] for r in records) / len(records), 2
        )
    }

    per_day = defaultdict(set)
    for r in records:
        per_day[r["timestamp"].date()].add(r["student_id"])

    trend_data = {
        day.strftime("%Y-%m-%d"): len(students)
        for day, students in sorted(per_day.items())
    }

    # -----------------------------
    # Default Insights (always safe)
    # -----------------------------
    insights = {
        "overview": f"{stats['unique_students']} unique students recorded across {len(trend_data)} days.",
        "trend": "Attendance remains generally stable with normal daily variation.",
        "action": "Review low-attendance days to improve engagement or scheduling."
    }

    # -----------------------------
    # Gemini AI Enhancement
    # -----------------------------
    if genai_client:
        try:
            prompt = f"""
                You are an AI analytics assistant for a university classroom attendance system.

                Context:
                This data comes from real-time facial recognition-based attendance.

                Attendance statistics:
                {stats}

                Daily attendance trend:
                {trend_data}

                Instructions:
                - Write a detailed but clear overview (2-3 sentences)
                - Explain attendance trends with reasoning (2-3 sentences)
                - Give a practical, actionable recommendation (2-3 sentences)
                - Avoid bullet points
                - Write in a professional, analytical tone
            """


            response = genai_client.models.generate_content(
                model="gemini-3-flash",
                contents=prompt
            )

            if response and response.text:
                lines = [l.strip() for l in response.text.split("\n") if l.strip()]
                if len(lines) >= 3:
                    insights["overview"] = lines[0]
                    insights["trend"] = lines[1]
                    insights["action"] = lines[2]

        except Exception as e:
            print("Gemini skipped:", e)

    return render_template(
        "analytics.html",
        stats=stats,
        trend_data=trend_data,
        insights=insights
    )

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

                    # REGISTER
                    if mode == "register" and student_id and student_name:
                        if not registration_done:
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

                    # DETECT / ATTENDANCE
                    if mode in ("detect", "attendance") and registered_students:
                        now = time.time()
                        if now - last_embed_time < 0.4:
                            continue
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

                            if mode == "attendance" and sid not in attendance_marked:
                                mark_attendance(
                                    sid,
                                    best_match["name"],
                                    float(best_score)
                                )
                                attendance_marked.add(sid)
                                status_message = "Attendance Posted"
                            elif mode == "detect":
                                status_message = ""

                            cv2.putText(
                                frame,
                                sid,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.8,
                                (0, 255, 0),
                                3
                            )
                        else:
                            status_message = (
                                "Attendance Not Posted"
                                if mode == "attendance"
                                else "Not Detected"
                            )

            if registration_done and registration_done_time:
                if time.time() - registration_done_time > 2:
                    status_message = ""
                    registration_done = False
                    registration_done_time = None

        except Exception as e:
            print("ERROR:", e)
            pass

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
