# 🎵 Doomscrolling Detector

A real-time doomscrolling detector that watches your eyes via webcam.  
Look down for **10 seconds** with no keyboard or mouse activity — and face the music.  
Look back up to make it stop.

---

## How it works

1. **Gaze tracking** — MediaPipe Face Mesh detects where your iris is pointing
2. **Input monitoring** — `pynput` listens for any mouse or keyboard activity in the background
3. **Timer** — a progress bar fills up while you look down with no input. Any activity resets it
4. **Punishment** — after 10 seconds, `music.mp3` starts playing on loop
5. **Redemption** — look back up and the music fades out

---

## Project structure

```
doomscrolling_detector/
├── main.py            # Entry point — runs the main loop
├── gaze.py            # MediaPipe gaze & eye detection + drawing helpers
├── input_tracker.py   # Mouse & keyboard activity tracking
├── music_player.py    # pygame music playback
├── config.py          # All tuneable settings in one place
├── requirements.txt   # Python dependencies
├── music.mp3          # Your punishment track (not included)
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/idaho73/doomscrolling_detector.git
cd doomscrolling_detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your music file**

Place an MP3 named `music.mp3` in the project root.  
*(We recommend [Never Gonna Give You Up](https://www.youtube.com/watch?v=dQw4w9WgXcQ) 🎵)*

**4. Run**
```bash
python main.py
```

The face landmarker model (~2 MB) will be downloaded automatically on first run.

---

## Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---|---|---|
| `ALERT_SECONDS` | `10.0` | Seconds of downward gaze before music triggers |
| `LOOK_DOWN_THRESH` | `0.60` | Iris ratio threshold for "looking down" |
| `EAR_THRESH` | `0.20` | Eye Aspect Ratio threshold for blink detection |
| `MUSIC_FADEOUT_MS` | `800` | Fade-out time in ms when music stops |
| `CAMERA_INDEX` | `0` | Webcam index (0 = default) |

---

## Requirements

- Python 3.8+
- Webcam
- Good lighting (works with glasses ✅)

---

## How the gaze detection works

The detector uses **MediaPipe Face Mesh** with 478 facial landmarks.  
Rather than looking for "eye shapes" like old Haar Cascade methods, it tracks the exact position of your **iris** within your eye opening — which is why it works reliably even through glasses.

The **iris vertical ratio** tells us where the iris sits between the top and bottom of the eye:
- `~0.6` → looking straight ahead
- `< 0.30` → looking down (triggering the detector)

Any mouse movement, click, or keypress resets the 10-second timer, so normal computer use won't get you rickrolled.

---

## License

MIT
