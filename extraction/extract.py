import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

N_LANDMARKS = 543
BLOCKS = [("face_landmarks", 0, 468), ("left_hand_landmarks", 468, 21),
          ("pose_landmarks", 489, 33), ("right_hand_landmarks", 522, 21)]


def _points(landmarks):
    if landmarks and isinstance(landmarks[0], list):
        landmarks = landmarks[0]
    return landmarks


def result_to_array(result):
    out = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for name, start, n in BLOCKS:
        lms = _points(getattr(result, name))
        if lms:
            out[start:start + n] = [[p.x, p.y, p.z] for p in lms[:n]]
    return out


def extract_video(video_path, model_path, target_fps=15):
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / target_fps))
    frames, i = [], 0
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(i * 1000 / fps))
                frames.append(result_to_array(result))
            i += 1
    cap.release()
    if not frames:
        return np.zeros((0, N_LANDMARKS, 3), dtype=np.float32)
    arr = np.stack(frames)
    h, w = rgb.shape[:2]
    arr[..., 0] *= w / h
    arr[..., 2] *= w / h
    return arr