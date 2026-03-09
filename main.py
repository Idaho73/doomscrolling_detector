"""
main.py — Doomscrolling Detector entry point.

Look down for 10 seconds with no mouse/keyboard activity
and face the music. Look back up to make it stop.

Usage:
    python main.py
"""

import cv2
import time
import os
import sys

import config
from gaze import GazeDetector, download_model, draw_timer_bar, draw_hud
from input_tracker import InputTracker
from music_player import MusicPlayer


def main():
    # Validate music file exists before starting
    if not os.path.exists(config.MUSIC_FILE):
        print(f"Error: '{config.MUSIC_FILE}' not found.")
        print("Place your music file in the project root and make sure it's named 'music.mp3'.")
        sys.exit(1)

    download_model()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        sys.exit(1)

    # Initialise modules
    gaze    = GazeDetector()
    inputs  = InputTracker()
    music   = MusicPlayer()

    inputs.start()
    music.start()

    # State
    blink_count  = 0
    was_closed   = False
    looking_down = False
    look_down_ts = None   # timestamp when current downward gaze streak started

    print("=" * 50)
    print("  Doomscrolling Detector — running")
    print(f"  Trigger: look down for {config.ALERT_SECONDS}s with no input")
    print("  Press Q to quit")
    print("=" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            now  = time.time()
            ts   = int(now * 1000)

            # Run gaze detection
            gaze.process(frame, ts)
            info = gaze.analyse(frame)

            # Blink counting
            both_closed = not info["l_open"] and not info["r_open"]
            if both_closed:
                if not was_closed:
                    blink_count += 1
                was_closed = True
            else:
                was_closed = False

            # ── Timer & music logic ───────────────────────────────────────
            if info["looking_down"]:
                idle_secs = inputs.seconds_since_input()

                if not looking_down:
                    look_down_ts = now                   # fresh streak
                else:
                    # If there was recent input, slide the start forward
                    effective_start = now - idle_secs
                    if effective_start > look_down_ts:
                        look_down_ts = effective_start

                elapsed = now - look_down_ts

                if elapsed >= config.ALERT_SECONDS and not music.playing:
                    music.play()

            else:
                if music.playing:
                    music.stop()
                look_down_ts = None

            looking_down = info["looking_down"]

            # ── Timer bar ─────────────────────────────────────────────────
            if looking_down and look_down_ts is not None:
                elapsed   = now - look_down_ts
                remaining = max(0.0, config.ALERT_SECONDS - elapsed)
                draw_timer_bar(frame, elapsed, config.ALERT_SECONDS, w, h)

                label     = "🎵 Never Gonna Give You Up!" if music.playing else f"Music in: {remaining:.1f}s"
                t_col     = (0, 200, 100) if inputs.seconds_since_input() < 1.5 else (0, 40, 255)
                cv2.putText(frame, label, (10, h - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, t_col, 1)

            # ── HUD ───────────────────────────────────────────────────────
            gaze_label = "LOOKING DOWN" if looking_down else "Looking ahead"
            gaze_color = (0, 80, 255)   if looking_down else (0, 220, 100)

            idle_secs   = inputs.seconds_since_input()
            kind        = inputs.last_input_kind()
            input_label = f"Input: {kind} ({idle_secs:.1f}s ago)" if kind else "Input: idle"
            input_color = (0, 210, 255) if idle_secs < 2 else (100, 100, 100)

            draw_hud(frame, gaze_label, gaze_color,
                     input_label, input_color,
                     blink_count, music.playing, w, h)

            cv2.imshow("Doomscrolling Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        music.stop_and_quit()
        inputs.stop()
        gaze.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended. Total blinks: {blink_count}")


if __name__ == "__main__":
    main()