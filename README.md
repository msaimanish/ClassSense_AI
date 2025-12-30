# ClassSense_AI

A real-time facial recognition–based classroom attendance system built to address proxy attendance, inefficiency, and scalability issues in large classrooms. The system uses computer vision and cloud technologies from Google to automatically detect, recognize, and record student attendance.

---

## Problem Statement

In many campuses, attendance is:
- Time-consuming
- Prone to proxy or misuse (QR sharing, manual roll calls)
- Difficult to manage in large classrooms with a large number of students

This project provides a hands-free, real-time attendance mechanism using facial recognition, designed specifically for on-campus deployment.

---

## Solution Overview

- Detects multiple faces in real time from a classroom camera
- Recognizes registered students using face embeddings
- Marks attendance automatically with confidence scores
- Stores attendance securely in the cloud
- Provides a web-based interface for teachers

---

## Machine Learning Approach

- Deep learning–based computer vision
- Metric learning using face embeddings (FaceNet-style)
- Pre-trained models for fast and reliable inference
- Real-time inference on live video streams

No retraining is required when new students are added.

---

## Google Technologies Used

ClassSense_AI extensively uses Google technologies:

- **Google MediaPipe**  
  Used for efficient and reliable multi-face detection in real-time video streams.

- **TensorFlow and Keras**  
  Used to generate face embeddings for identity representation and comparison.

- **Google Cloud Firestore**  
  Acts as the primary cloud database for storing:
  - Registered student face embeddings
  - Attendance records
  - Confidence scores and timestamps

- **Firebase Admin SDK**  
  Enables secure server-side access to Firestore from the Flask backend.

- **Google Gemini AI**  
  Used in the Analytics Dashboard to:
  - Generate natural-language summaries of attendance trends
  - Analyze daily and overall attendance patterns
  - Provide AI-driven insights such as participation consistency
  - Assist in explaining trends in a human-readable form

---

## Features

- Live camera feed with face bounding boxes
- Student face registration
- Confidence-based attendance marking
- Attendance registry with search and date filters
- Web-based interface (no mobile application required)

---

## Project Structure

```text
ClassSense_AI/
├── app.py
├── db/
│   └── firestore.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── attendance.html
│   └── analytics.html
├── static/
│   └── styles.css
├── requirements.txt
├── README.md
└── serviceAccountKey.json  (not committed)
```
---

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/msaimanish/ClassSense_AI
cd ClassSense_AI


### 3. Install Dependencies
pip install -r requirements.txt


### 4. Firebase Setup
1. Create a Firebase project
2. Enable Cloud Firestore
3. Generate a service account key
4. Place the key file in the project root as:
serviceAccountKey.json


This file is intentionally excluded from version control.

---

## Running the Application
python app.py

## Intended Use Case

- On-campus deployment with camera access
- Large classroom environments
- Institutions using Google cloud infrastructure

---

## Future Enhancements

- Advanced attendance analytics dashboard
- Attendance trend prediction
- Anomaly detection for low attendance days
- Google Classroom integration
- PDF and CSV attendance report generation

---

## License

This project is intended for educational and hackathon use.