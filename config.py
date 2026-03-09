# ── Doomscrolling Detector — Configuration ────────────────────────────────────
# Tweak these values to adjust the detector's behaviour.

# --- Gaze ---
# Iris vertical ratio threshold (0.0 = top of eye, 1.0 = bottom)
# Lower value = more sensitive to downward gaze
LOOK_DOWN_THRESH = 0.30

# Eye Aspect Ratio threshold — below this the eye is considered closed
EAR_THRESH = 0.20

# --- Timer ---
# Seconds of looking down with no input before music triggers
ALERT_SECONDS = 10.0

# Seconds between repeated music triggers (after looking up and back down)
MUSIC_COOLDOWN = 5.0

# --- Input ---
# Seconds of mouse/keyboard inactivity before the timer is no longer reset
INPUT_IDLE_WINDOW = 1.5

# --- Music ---
MUSIC_FILE = "music.mp3"

# Fade-out duration in milliseconds when looking back up
MUSIC_FADEOUT_MS = 800

# --- Camera ---
CAMERA_INDEX = 1      # 0 = default webcam 1 = USB webcam

# --- MediaPipe ---
MAX_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)