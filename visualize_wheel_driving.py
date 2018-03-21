import numpy as np
import pandas as pd
import pygame
import glob
from config import VisualizeConfig
import scipy.misc

BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)
ORANGE = (255, 165, 0)

'''config = VisualizeConfig()
preds = pd.read_csv(config.pred_path)
true = pd.read_csv(config.true_path)
filenames = glob.glob(config.img_path)'''
filenames = []
with open('CH2_final_evaluation.txt', 'r') as f:
    for line in f.readlines():
        filenames.append('center/' + line.strip().split(',')[0] + '.png')

gt = []
ours = []
pred = []
with open('udacity_compare.txt', 'r') as h:
    for line in h.readlines():
        gt.append(float(line.strip().split(',')[0]))
        ours.append(float(line.strip().split(',')[1]))
        pred.append(float(line.strip().split(',')[2]))


pygame.init()
size = (640, 480)
pygame.display.set_caption("Data viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
myfont = pygame.font.SysFont("monospace", 15)

for i in range(670, 870):
#for i in range(len(filenames)):
    angle = 0.7 * ours[i-670] + 0.3 * ours[i-669] # preds["steering_angle"].iloc[i] # radians
    true_angle = 0.7 * gt[i-670] + 0.3 * gt[i-669] # true["steering_angle"].iloc[i] # radians
    base_angle = 0.7 * pred[i-670] + 0.3 * pred[i-669]
    
    # add image to screen
    img = pygame.image.load(filenames[i])
    screen.blit(img, (0, 0))
    
    # add text
    '''pred_txt = myfont.render("Prediction:" + str(round(angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    true_txt = myfont.render("True angle:" + str(round(true_angle* 57.2958, 3)), 1, (255,255,0)) # angle in degrees
    screen.blit(pred_txt, (10, 280))
    screen.blit(true_txt, (10, 300))'''

    # draw steering wheel
    radius = 100
    pygame.draw.circle(screen, WHITE, [320, 480], radius, 4) 

    # draw cricle for true angle
    x = radius * np.cos(np.pi/2 + true_angle * 3.1415 / 180.0)
    y = radius * np.sin(np.pi/2 + true_angle * 3.1415 / 180.0)
    # pygame.draw.circle(screen, WHITE, [320 + int(x), 300 - int(y)], 7)
    pygame.draw.line(screen, GREEN, [320, 480], [320 + int(x), 480 - int(y)], 2)    

    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + angle * 3.1415 / 180.0)
    y = radius * np.sin(np.pi/2 + angle * 3.1415 / 180.0)
    # pygame.draw.circle(screen, BLACK, [320 + int(x), 300 - int(y)], 5) 
    pygame.draw.line(screen, RED, [320, 480], [320 + int(x), 480 - int(y)], 2)

    # draw cricle for predicted angle
    x = radius * np.cos(np.pi/2 + base_angle * 3.1415 / 180.0)
    y = radius * np.sin(np.pi/2 + base_angle * 3.1415 / 180.0)
    # pygame.draw.circle(screen, BLACK, [320 + int(x), 300 - int(y)], 5) 
    pygame.draw.line(screen, ORANGE, [320, 480], [320 + int(x), 480 - int(y)], 2)
    scipy.misc.imsave('/home/cardwing/Downloads/self-driving-car-master/steering-models/community-models/rambo/demo/' + str(i) + '.png', np.rot90(pygame.surfarray.array3d(screen), 3))
    #pygame.display.update()
    pygame.display.flip()
    
