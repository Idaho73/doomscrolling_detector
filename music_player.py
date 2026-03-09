"""
music_player.py
Handles loading and playback of the punishment track via pygame.
"""

import pygame
import config


class MusicPlayer:
    """
    Wraps pygame.mixer for simple play/stop control.
    Call start() before using, stop_and_quit() when done.
    """

    def __init__(self):
        self.playing = False

    def start(self):
        pygame.mixer.init()

    def play(self):
        if not self.playing:
            pygame.mixer.music.load(config.MUSIC_FILE)
            pygame.mixer.music.play(-1)   # loop indefinitely
            self.playing = True

    def stop(self):
        if self.playing:
            pygame.mixer.music.fadeout(config.MUSIC_FADEOUT_MS)
            self.playing = False

    def stop_and_quit(self):
        self.stop()
        pygame.mixer.quit()