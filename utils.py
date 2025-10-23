import cv2
import numpy as np
from math import cos, sin
import mediapipe as mp
import joblib
from collections import deque
import tempfile
import os

# Define the key landmarks and load models
SELECTED_LANDMARKS = [1, 33, 263, 10, 234, 454, 152]
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Load the pre-trained models
try:
    pitch_model = joblib.load('model_pitch.pkl')
    yaw_model = joblib.load('model_yaw.pkl')
    roll_model = joblib.load('model_roll.pkl')
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    raise

class PoseSmoother:
    def __init__(self, window=5):
        self.pitch_hist = deque(maxlen=window)
        self.yaw_hist = deque(maxlen=window)
        self.roll_hist = deque(maxlen=window)

    def update(self, pitch, yaw, roll):
        self.pitch_hist.append(pitch)
        self.yaw_hist.append(yaw)
        self.roll_hist.append(roll)
        smooth_pitch = np.mean(self.pitch_hist)
        smooth_yaw = np.mean(self.yaw_hist)
        smooth_roll = np.mean(self.roll_hist)
        return smooth_pitch, smooth_yaw, smooth_roll

def normalize_features(landmarks):
    """Normalizes face landmarks based on nose and eye positions."""
    if not landmarks:
        return None
    
    landmarks_dict = {lm_id: landmarks[lm_id] for lm_id in SELECTED_LANDMARKS}

    nose = landmarks_dict[1]
    eye_left = landmarks_dict[33]
    eye_right = landmarks_dict[263]

    eye_dist = np.sqrt((eye_right.x - eye_left.x)**2 + (eye_right.y - eye_left.y)**2)
    if eye_dist == 0:
        eye_dist = 1e-6

    feats = []
    for idx in SELECTED_LANDMARKS:
        lm = landmarks_dict[idx]
        feats.append((lm.x - nose.x) / eye_dist)
        feats.append((lm.y - nose.y) / eye_dist)
        
    return np.array(feats).reshape(1, -1)

def draw_pose_axes(img, pitch, yaw, roll, nose_point, length=100):
    """Draws 3D pose axes on an image."""
    pitch_rad = np.radians(-pitch)
    yaw_rad = np.radians(yaw)
    roll_rad = np.radians(roll)

    h, w, _ = img.shape
    cx, cy = int(nose_point[0] * w), int(nose_point[1] * h)

    R_x = np.array([[1, 0, 0], [0, cos(pitch_rad), -sin(pitch_rad)], [0, sin(pitch_rad), cos(pitch_rad)]])
    R_y = np.array([[cos(yaw_rad), 0, sin(yaw_rad)], [0, 1, 0], [-sin(yaw_rad), 0, cos(yaw_rad)]])
    R_z = np.array([[cos(roll_rad), -sin(roll_rad), 0], [sin(roll_rad), cos(roll_rad), 0], [0, 0, 1]])
    R = R_z @ R_y @ R_x

    axis_x = R @ np.array([length, 0, 0])
    axis_y = R @ np.array([0, -length, 0])
    axis_z = R @ np.array([0, 0, -length])

    def project(pt): return int(cx + pt[0]), int(cy + pt[1])

    x2, y2, z2 = project(axis_x), project(axis_y), project(axis_z)

    cv2.arrowedLine(img, (cx, cy), x2, (0, 0, 255), 3)
    cv2.arrowedLine(img, (cx, cy), y2, (0, 255, 0), 3)
    cv2.arrowedLine(img, (cx, cy), z2, (255, 0, 0), 3)

    return img

def predict_pose(image):
    """Processes a single image to predict pose and draw axes."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, "No face detected in the image."

    face_landmarks = results.multi_face_landmarks[0]
    landmarks_list = list(face_landmarks.landmark)
    feats = normalize_features(landmarks_list)
    
    pitch = pitch_model.predict(feats)[0]
    yaw = yaw_model.predict(feats)[0]
    roll = roll_model.predict(feats)[0]
    
    nose_point = (face_landmarks.landmark[1].x, face_landmarks.landmark[1].y)
    output_image = draw_pose_axes(image.copy(), pitch, yaw, roll, nose_point)
    
    return output_image, None

def process_video(video_path, output_path, frame_skip=1, smooth_window=2):
    """Processes a video file to predict pose and draw axes on each frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Error opening video file:"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    smoother = PoseSmoother(window=smooth_window)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % frame_skip != 0:
            out.write(frame)
            continue
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            nose = face_landmarks.landmark[1]
            eye_left, eye_right = face_landmarks.landmark[33], face_landmarks.landmark[263]

            eye_dist = np.sqrt((eye_right.x - eye_left.x)**2 + (eye_right.y - eye_left.y)**2)
            if eye_dist == 0: 
                eye_dist = 1e-6

            feats = []
            for idx in SELECTED_LANDMARKS:
                lm = face_landmarks.landmark[idx]
                feats.append((lm.x - nose.x) / eye_dist)
                feats.append((lm.y - nose.y) / eye_dist)

            feats = np.array(feats).reshape(1, -1)

            # Predictions
            pitch = pitch_model.predict(feats)[0]
            yaw   = yaw_model.predict(feats)[0]
            roll  = roll_model.predict(feats)[0]

            # Smooth predictions
            pitch, yaw, roll = smoother.update(pitch, yaw, roll)

            nose_point = (nose.x, nose.y)
            frame = draw_pose_axes(frame, pitch, yaw, roll, nose_point)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Processed video saved to: {output_path}")

    return output_path, None
