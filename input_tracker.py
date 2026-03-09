"""
input_tracker.py
Tracks mouse and keyboard activity in background threads.
"""

import time
import threading
from pynput import mouse, keyboard


class InputTracker:
    """
    Listens to mouse movement, clicks, and keyboard presses.
    Thread-safe — can be queried from any thread.
    """

    def __init__(self):
        self._lock         = threading.Lock()
        self.last_activity = time.time()   # initialised to now
        self.last_kind     = ""
        self._mouse_listener    = None
        self._keyboard_listener = None

    def register(self, kind: str):
        with self._lock:
            self.last_activity = time.time()
            self.last_kind     = kind

    def seconds_since_input(self) -> float:
        with self._lock:
            return time.time() - self.last_activity

    def last_input_kind(self) -> str:
        with self._lock:
            return self.last_kind

    def start(self):
        self._mouse_listener = mouse.Listener(
            on_move=lambda x, y: self.register("Mouse move"),
            on_click=lambda x, y, b, p: self.register("Mouse click"),
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=lambda k: self.register("Keyboard"),
        )
        self._mouse_listener.start()
        self._keyboard_listener.start()

    def stop(self):
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._keyboard_listener:
            self._keyboard_listener.stop()