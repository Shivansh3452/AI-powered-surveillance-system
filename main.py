import cv2
import mediapipe as mp
import time
import os
import csv
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

# --- CONFIGURATION ---
VIDEO_SOURCE = 0  # Try 0 first. If black screen, change to 1.
ALERT_COOLDOWN = 2  # Seconds between saving evidence
EVIDENCE_FOLDER = "evidence_locker"
LOG_FILE = "security_log.csv"
MODEL_FILE = "harassment_model.pkl"

# --- SETUP FILES ---
if not os.path.exists(EVIDENCE_FOLDER):
    os.makedirs(EVIDENCE_FOLDER)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event Type", "Confidence", "Image Path"])

# --- AI ENGINE ---
class HarassmentDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.status = "System Online"
        self.confidence = 0.0
        self.mode = "Heuristic" # Default to simple math
        
        # Try to load the ML Model
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.mode = "ML_Model"
                print(f"✅ AI Model Loaded: {MODEL_FILE}")
            except:
                print("⚠️ Error loading model. Switching to Basic Mode.")
        else:
            print("ℹ️ No model found. Using Basic Mode (Wrist vs Nose).")

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        annotated_frame = frame.copy()
        threat_detected = False
        h, w, c = frame.shape

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(annotated_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # --- METHOD A: MACHINE LEARNING (If model exists) ---
            if self.mode == "ML_Model":
                try:
                    # Extract coordinates
                    pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
                    
                    # Predict
                    X = pd.DataFrame([pose_row]) 
                    prediction = self.model.predict(X)[0]
                    probs = self.model.predict_proba(X)[0]
                    
                    # Get confidence of the predicted class
                    class_idx = list(self.model.classes_).index(prediction)
                    confidence = probs[class_idx]
                    
                    # Filter: Only alert if confidence is high (> 0.7)
                    if prediction == "Attack" and confidence > 0.7:
                        threat_detected = True
                        self.status = "CRITICAL: HARASSMENT (AI)"
                        self.confidence = round(confidence, 2)
                        cv2.putText(annotated_frame, f"AI ALERT: {int(confidence*100)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        self.status = "Monitoring (AI)"
                        self.confidence = round(confidence, 2)

                except Exception as e:
                    print(f"Prediction Error: {e}")

            # --- METHOD B: HEURISTIC (Backup Logic) ---
            # Used if ML fails or if no model is loaded
            else:
                nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y
                left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
                right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
                
                # Visual Line
                nose_pixel = int(nose_y * h)
                cv2.line(annotated_frame, (0, nose_pixel), (w, nose_pixel), (255, 255, 0), 2)

                # Logic: Hands higher than nose
                if left_wrist_y < nose_y or right_wrist_y < nose_y:
                    threat_detected = True
                    self.status = "CRITICAL: AGGRESSIVE POSE"
                    self.confidence = 0.90
                    cv2.putText(annotated_frame, "AGGRESSIVE POSE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    self.status = "Monitoring"
                    self.confidence = 0.1

        return annotated_frame, threat_detected, self.status, self.confidence

# --- SERVER ---
app = FastAPI()
detector = HarassmentDetector()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

latest_alert = {"timestamp": None, "status": "Initializing...", "confidence": 0.0}

def save_evidence(frame, status, conf):
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{EVIDENCE_FOLDER}/alert_{timestamp_str}.jpg"
    cv2.imwrite(filename, frame)
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([timestamp_str, status, conf, filename])
    print(f"📸 Evidence Saved: {filename}")

def generate_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    last_alert_time = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        processed_frame, is_threat, status, conf = detector.detect(frame)
        
        current_time = time.time()
        
        # Global Status Update for Frontend
        global latest_alert
        if is_threat:
            latest_alert = {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "confidence": conf
            }
            # Save Evidence (with cooldown)
            if (current_time - last_alert_time > ALERT_COOLDOWN):
                save_evidence(processed_frame, status, conf)
                last_alert_time = current_time
        else:
            # Send 'Normal' status so frontend stops flashing red
            latest_alert["status"] = "Normal"
            latest_alert["confidence"] = conf

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/alerts")
def get_alerts():
    return JSONResponse(content=latest_alert)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)