"""
gaze.py
Handles MediaPipe face landmark detection, gaze direction,
eye open/close state, and all OpenCV drawing helpers.
"""

import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os

import config

# ── MediaPipe Tasks ───────────────────────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
FaceLandmarker       = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpts   = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

# ── Landmark index groups ─────────────────────────────────────────────────────
LEFT_EYE      = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE     = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS     = [474, 475, 476, 477]
RIGHT_IRIS    = [469, 470, 471, 472]
LEFT_IRIS_CTR = 473;  RIGHT_IRIS_CTR = 468
LEFT_EYE_TOP  = 386;  LEFT_EYE_BOT   = 374
RIGHT_EYE_TOP = 159;  RIGHT_EYE_BOT  = 145

EAR_IDX = [0, 4, 3, 5, 1, 2]


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _get_pt(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)

def _pts_array(lm, indices, w, h):
    return np.array([_get_pt(lm, i, w, h) for i in indices], dtype=np.int32)

def _eye_aspect_ratio(p):
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    return (A + B) / (2.0 * C + 1e-6)

def _vertical_iris_ratio(lm, ctr, top, bot, w, h):
    _, iy = _get_pt(lm, ctr, w, h)
    _, ty = _get_pt(lm, top, w, h)
    _, by = _get_pt(lm, bot, w, h)
    span  = by - ty
    return (iy - ty) / span if span > 0 else 0.5


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_eye_contour(frame, points, color):
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

def draw_iris(frame, points, color):
    center = points.mean(axis=0).astype(int)
    radius = max(1, int(np.linalg.norm(points[0] - points[2]) / 2))
    cv2.circle(frame, tuple(center), radius, color, 1)
    cv2.circle(frame, tuple(center), 2,      color, -1)

def draw_timer_bar(frame, elapsed, total, w, h):
    ratio  = min(elapsed / total, 1.0)
    filled = int(w * ratio)
    bar_y  = h - 10
    cv2.rectangle(frame, (0, bar_y), (w, h), (30, 30, 30), -1)
    if filled > 0:
        if ratio < 0.5:   color = (0, 200, 100)
        elif ratio < 0.8: color = (0, 140, 255)
        else:             color = (0, 40,  255)
        cv2.rectangle(frame, (0, bar_y), (filled, h), color, -1)

def draw_hud(frame, gaze_label, gaze_color, input_label, input_color,
             blink_count, music_playing, w, h):
    hud = frame.copy()
    bg_color = (0, 0, 100) if music_playing else (15, 15, 15)
    cv2.rectangle(hud, (0, 0), (w, 78), bg_color, -1)
    cv2.addWeighted(hud, 0.55, frame, 0.45, 0, frame)

    if music_playing:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 200), 4)

    cv2.putText(frame, gaze_label,  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, gaze_color,  2)
    cv2.putText(frame, input_label, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, input_color, 1)
    cv2.putText(frame, f"Blinks: {blink_count}", (w - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)


# ── Model download ────────────────────────────────────────────────────────────
def download_model():
    if not os.path.exists(config.MODEL_PATH):
        print("Downloading face landmarker model (~2 MB)...")
        urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
        print("Model ready!")


# ── GazeDetector ─────────────────────────────────────────────────────────────
class GazeDetector:
    """
    Wraps the MediaPipe FaceLandmarker in LIVE_STREAM mode.
    Call process(frame) each loop iteration, then read .result.
    """

    def __init__(self):
        self._latest = None
        options = FaceLandmarkerOpts(
            base_options=BaseOptions(model_asset_path=config.MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=config.MAX_FACES,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            result_callback=self._on_result,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def _on_result(self, result, output_image, timestamp_ms):
        self._latest = result

    def process(self, frame, timestamp_ms: int):
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        self._landmarker.detect_async(mp_img, timestamp_ms)

    def analyse(self, frame):
        """
        Returns a dict with gaze/eye info drawn onto frame.
        Keys: looking_down, blink, l_open, r_open, avg_ratio
        """
        result = {"looking_down": False, "blink": False,
                  "l_open": True, "r_open": True, "avg_ratio": 0.5}

        if not self._latest or not self._latest.face_landmarks:
            return result

        h, w = frame.shape[:2]
        lm    = self._latest.face_landmarks[0]

        l_pts  = _pts_array(lm, LEFT_EYE,  w, h)
        r_pts  = _pts_array(lm, RIGHT_EYE, w, h)
        l_open = _eye_aspect_ratio(l_pts[EAR_IDX]) > config.EAR_THRESH
        r_open = _eye_aspect_ratio(r_pts[EAR_IDX]) > config.EAR_THRESH

        l_ratio   = _vertical_iris_ratio(lm, LEFT_IRIS_CTR,  LEFT_EYE_TOP,  LEFT_EYE_BOT,  w, h)
        r_ratio   = _vertical_iris_ratio(lm, RIGHT_IRIS_CTR, RIGHT_EYE_TOP, RIGHT_EYE_BOT, w, h)
        avg_ratio = (l_ratio + r_ratio) / 2.0
        now_down  = avg_ratio < config.LOOK_DOWN_THRESH and (l_open or r_open)

        # Draw contours & iris
        draw_eye_contour(frame, l_pts, (0, 220, 100) if l_open else (30, 60, 220))
        draw_eye_contour(frame, r_pts, (0, 220, 100) if r_open else (30, 60, 220))
        iris_col = (0, 60, 255) if now_down else (255, 200, 50)
        draw_iris(frame, _pts_array(lm, LEFT_IRIS,  w, h), iris_col)
        draw_iris(frame, _pts_array(lm, RIGHT_IRIS, w, h), iris_col)

        result.update(looking_down=now_down, l_open=l_open,
                      r_open=r_open, avg_ratio=avg_ratio)
        return result

    def close(self):
        self._landmarker.close()