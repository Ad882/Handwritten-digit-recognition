# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:24:48 2023

@author: Adrien M.
"""

import pygame, sys
from pygame.locals import QUIT, MOUSEMOTION, MOUSEBUTTONDOWN, MOUSEBUTTONUP, K_r
import numpy as np
import tensorflow as tf
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PREDICT = True
IMAGESAVE = False
MODEL = tf.keras.models.load_model("bestmodel.h5")   # Load the model trained in the file model.py.
LABELS = {0 : "Zero", 1 : "One", 2 : "Two", 3 : "Three", 4 : "Four", 5 : "Five", 6 : "Six", 7 : "Seven", 8 : "Eight", 9 : "Nine"}

pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 20)   # Load a font for text rendering.
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))   # Create the display surface.
WHITE_INT = DISPLAYSURF.map_rgb(WHITE)   # Get the integer representation of the white color.
pygame.display.set_caption("Digit Board")   # Set the window caption.

iswriting = False   # Flag to track if the mouse is drawing.
number_xcord = []   # List to store x-coordinates of drawn points.
number_ycord = []   # List to store y-coordinates of drawn points.
image_count = 1   # Counter for saving images.

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
            
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)   # Draw white circles as the mouse moves.
            
            number_xcord.append(xcord)   # Store x-coordinate of the drawn point.
            number_ycord.append(ycord)   # Store y-coordinate of the drawn point.
            
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True   # Start writing when the mouse button is pressed.
            
        if event.type == MOUSEBUTTONUP:
            iswriting = False   # Stop writing when the mouse button is released.
            number_xcord = sorted(number_xcord)   # Sort the x-coordinates.
            number_ycord = sorted(number_ycord)   # Sort the y-coordinates.
            
            # Determine the bounding box for the drawn digit.
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)

            number_xcord = []   # Reset x-coordinate list.
            number_ycord = []   # Reset y-coordinate list.
            
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)   # Extract the drawn digit from the display surface and convert to numpy array.
            
            if IMAGESAVE:
                cv2.imwrite("digit.png")
                image_count += 1
            
            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))   # Resize the drawn digit.
                image = np.pad(image, (10, 10), 'constant', constant_values=0)   # Add padding.
                image = cv2.resize(image, (28, 28)) / 255   # Resize and normalize image.
                
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])  # Use the loaded model to predict the digit.
                
                textSurface = FONT.render(label, True, RED, WHITE)  # Create a text surface with the prediction label and position it.
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_min_y
                
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)   # Draw a rectangle around the digit and display the prediction label.
                DISPLAYSURF.blit(textSurface, textRecObj)


    pygame.display.update()   # Update the display.
    
    keys = pygame.key.get_pressed()   # Get the state of all keyboard keys.
    if keys[K_r]:   
        DISPLAYSURF.fill(BLACK)   # If the key 'r' is pressed, fill the display surface with black, effectively clearing it. 
