import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# -------------------------
# INITIALIZE FIREBASE (ONCE)
# -------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------
# STUDENTS COLLECTION
# -------------------------
def student_exists(student_id):
    return db.collection("students").document(student_id).get().exists


def save_student(student_id, name, embedding):
    db.collection("students").document(student_id).set({
        "name": name,
        "embedding": embedding,   # list[float]
        "created_at": firestore.SERVER_TIMESTAMP
    })


def get_all_students():
    docs = db.collection("students").stream()
    students = []
    for doc in docs:
        data = doc.to_dict()
        students.append({
            "id": doc.id,
            "name": data["name"],
            "embedding": data["embedding"]
        })
    return students

# -------------------------
# ATTENDANCE COLLECTION
# -------------------------
def mark_attendance(student_id, name, confidence):
    db.collection("attendance").add({
        "student_id": student_id,
        "name": name,
        "confidence": float(confidence),
        "timestamp": firestore.SERVER_TIMESTAMP
    })


def get_attendance():
    docs = (
        db.collection("attendance")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .stream()
    )
    return [doc.to_dict() for doc in docs]
