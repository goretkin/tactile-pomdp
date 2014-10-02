import pygame
from IPython.lib.inputhook import stdin_ready


class PyGame_InputHook():
    def __init__(self,callback):
        self.clock = pygame.time.Clock()
        self.callback = callback
        self.hz = 30

    def inputhook_pygame(self):
        self.clock.tick(self.hz) #if we broke out of the loop, ensure loop doesn't start back up too soon 
        while True:
            self.callback()
            if stdin_ready():
                break
            else:
                self.clock.tick(self.hz)
        return 0