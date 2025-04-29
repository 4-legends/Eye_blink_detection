import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import os
import subprocess
import tkinter as tk
from tkinter import messagebox

# Constants
INITIAL_BLINK_THRESHOLD = 0.25  # Initial threshold for eye aspect ratio
MIN_BLINKS_PER_MIN = 15  # Minimum number of blinks per minute
ALERT_INTERVAL = 60  # Alert interval in seconds
CONSECUTIVE_FRAMES = 2  # Number of consecutive frames below threshold to count as a blink
ADAPTIVE_THRESHOLD_WINDOW = 30  # Number of frames to consider for adaptive threshold
ADAPTIVE_THRESHOLD_PERCENTAGE = 0.85  # Percentage for adaptive threshold
BLINK_COOLDOWN = 10  # Number of frames to wait before detecting next blink

def eye_aspect_ratio(eye):
    # Calculate vertical distances
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    # Calculate horizontal distance
    C = distance.euclidean(eye[0], eye[3])
    # Calculate eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def show_popup():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showwarning("Blink Alert", "Please blink more often!")
    root.destroy()

def play_alert():
    # Play alert sound using aplay (Linux)
    try:
        subprocess.run(['aplay', '-q', '/usr/share/sounds/alsa/Front_Center.wav'], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        # If aplay fails, try using spd-say (text-to-speech)
        try:
            subprocess.run(['spd-say', 'Please blink more often'], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print("\a")  # Fallback to terminal bell
    
    # Show popup notification
    show_popup()

def main():
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    # Initialize variables
    blink_count = 0
    start_time = time.time()
    last_alert_time = time.time()
    eye_closed = False
    consecutive_frames = 0
    ear_history = []  # Store recent EAR values for adaptive thresholding
    current_threshold = INITIAL_BLINK_THRESHOLD
    last_ear = None  # Store the last EAR value
    cooldown_counter = 0  # Counter for blink cooldown
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale and apply histogram equalization
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect faces
        faces = detector(gray)
        
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # Get eye landmarks
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            
            # Calculate eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Update EAR history and calculate adaptive threshold
            ear_history.append(ear)
            if len(ear_history) > ADAPTIVE_THRESHOLD_WINDOW:
                ear_history.pop(0)
                # Set threshold to 85% of the average EAR
                current_threshold = np.mean(ear_history) * ADAPTIVE_THRESHOLD_PERCENTAGE
            
            # Handle blink cooldown
            if cooldown_counter > 0:
                cooldown_counter -= 1
                eye_closed = True
            else:
                eye_closed = False
            
            # Detect blink using both threshold and relative change
            if last_ear is not None and cooldown_counter == 0:
                ear_change = (last_ear - ear) / last_ear  # Calculate relative change
                if (ear < current_threshold or ear_change > 0.2) and not eye_closed:
                    consecutive_frames += 1
                    if consecutive_frames >= CONSECUTIVE_FRAMES:
                        blink_count += 1
                        eye_closed = True
                        cooldown_counter = BLINK_COOLDOWN
                else:
                    consecutive_frames = 0
            
            last_ear = ear  # Store current EAR for next iteration
            
            # Draw eye contours and threshold line
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
            cv2.putText(frame, f"Threshold: {current_threshold:.3f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Cooldown: {cooldown_counter}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Calculate time elapsed
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Check blink rate every minute
        if elapsed_time >= 60:
            blink_rate = blink_count / (elapsed_time / 60)
            if blink_rate < MIN_BLINKS_PER_MIN and (current_time - last_alert_time) >= ALERT_INTERVAL:
                play_alert()
                last_alert_time = current_time
                print(f"Alert: Blink rate is {blink_rate:.1f} blinks per minute (below minimum of {MIN_BLINKS_PER_MIN})")
            
            # Reset counters
            blink_count = 0
            start_time = current_time
        
        # Display blink count and rate
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Rate: {blink_count/(elapsed_time/60):.1f}/min", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("Eye Blink Detection", frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 