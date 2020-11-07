import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
pygame.init()
pygame.display.set_mode((1,1))

while 1:
    events = pygame.event.get()
    for e in events:
        pass