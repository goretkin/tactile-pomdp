from Box2D import *
import pygame, sys, math, random
from pygame.locals import *
 
# set up pygame-box2d constants
box_size=1.8
pillar_height=15.0
pygame_box2d_ratio=10.0
ground_height=10.0
pygame_screen_x=500
pygame_screen_y=400
 
# set up pygame
pygame.init()
  
# set up the window
windowSurface = pygame.display.set_mode((pygame_screen_x, pygame_screen_y))
pygame.display.set_caption('Box2D and Pygame demo!')
  
#setup boxes list
  
boxeslist=[]
  
# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class myContactListener(b2ContactListener):
    def __init__(self):
        b2ContactListener.__init__(self)
    def BeginContact(self, contact):
        print contact
    def EndContact(self, contact):
        pass
    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass

# set up  box2d world
world=b2World(contactListener=myContactListener())

# ground body
groundBody=world.CreateStaticBody(
    position=(25,ground_height/2.0),
    shapes=b2PolygonShape(box=(25,ground_height/2.0)),
    )
 
def create_dynamic_box(xpos):
    body=world.CreateDynamicBody(position=(xpos/pygame_box2d_ratio,40))
    box=body.CreatePolygonFixture(box=(box_size/2.0,box_size/2.0), density=1, friction=0.3)
    return body

world.CreateKinematicBody(position=(40,40))

def draw_dynamic_body(box):
    #rotate surf by DEGREE amount degrees
    rotatedredbox =  pygame.transform.rotozoom(redbox, math.degrees(box.angle),1)
    rotatedredbox.set_colorkey(0)
    windowSurface.blit(rotatedredbox, ((box.position.x*pygame_box2d_ratio)-((box_size/2.0)*pygame_box2d_ratio),pygame_screen_y-(box.position.y*pygame_box2d_ratio)-((box_size/2.0)*pygame_box2d_ratio)))
 

 
# load image
#redbox=pygame.image.load("d:\\redbox.png").convert()

#make falling squares
_a = box_size*pygame_box2d_ratio
redbox=pygame.Surface((_a,_a),)
pygame.draw.rect(redbox,RED,redbox.get_rect())

# Prepare for simulation. Typically we use a time step of 1/60 of a
# second (60Hz) and 6 velocity/2 position iterations. This provides a
# high quality simulation in most game scenarios.
timeStep = 1.0 / 300
vel_iters, pos_iters = 3, 1
  
# This is our little animation loop.
ev = pygame.event.poll()
while ev.type!=pygame.QUIT:
    ev = pygame.event.poll()
    keys=pygame.key.get_pressed()
    # create random boxes
    if random.randint(1,200)==2:
        boxeslist.append(create_dynamic_box(random.randint(1,500)))
  
    # Instruct the world to perform a single step of simulation. It is
    # generally best to keep the time step and iterations fixed.
    world.Step(timeStep, vel_iters, pos_iters)
  
    # Clear applied body forces. We didn't apply any forces, but you
    # should know about this function.
    world.ClearForces()
  
    # clean screen
    windowSurface.fill(BLACK)
  
    # draw ground
    pygame.draw.rect(windowSurface, GREEN, (0, pygame_screen_y-(ground_height*pygame_box2d_ratio), pygame_screen_x,pygame_screen_y ))
  
    for box in boxeslist:
        draw_dynamic_body(box)
    pygame.display.flip()
pygame.quit()