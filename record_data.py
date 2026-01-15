import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# --- CONFIGURATION ---
FILE_NAME = "training_data.csv"
CLASSES = ["Normal", "Attack"] 

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Create CSV file with headers if it doesn't exist
if not os.path.exists(FILE_NAME):
    landmarks = ['class']
    for val in range(1, 33+1):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    
    with open(FILE_NAME, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

print("--- CONTROLS ---")
print("Press 'n' to record NORMAL behavior")
print("Press 'a' to record ATTACK/HARASSMENT behavior")
print("Press 'q' to QUIT")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process Frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    # Draw Skeleton
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Show status
    cv2.putText(frame, "Hold 'n' for Normal | 'a' for Attack", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Data Collector', frame)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    
    # RECORD DATA
    if results.pose_landmarks:
        # Check which key is pressed
        target_class = ""
        if k == ord('n'):
            target_class = "Normal"
        elif k == ord('a'):
            target_class = "Attack"
            
        if target_class != "":
            # Extract Pose landmarks
            pose_row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten())
            
            # Append class name + coordinates
            row = [target_class] + pose_row
            
            # Write to CSV
            with open(FILE_NAME, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
            print(f"Recorded: {target_class}")

cap.release()
cv2.destroyAllWindows()