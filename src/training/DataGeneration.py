import cv2
import numpy as np
import copy
import time
from utils.drawobject import draw_object
import random
import os



def addLabel(highest, lowest, leftmost, rightmost, color, shape):
    # add labels
    label = open(str(color) + "_" + str(shape) + ".txt", "w")
    height = lowest - highest
    width = rightmost - leftmost
    y_center = (lowest + highest)/2
    x_center = (rightmost + leftmost)/2
    label.write("0 " + x_center.astype(str) + " " + y_center.astype(str) + " " + width.astype(str) + " " + height.astype(str))
    label.close
    
    
def main():
    # creates images with different shapes and colors
    
    # colors
    yellow = ["yellow", [0, 255, 255]]
    white = ["white", [255,255,255]]
    black = ["black", [0,0,0]]
    red = ["red", [0,0, 255]]
    blue = ["blue", [255,0,0]]
    green = ["green", [0,128,0]]
    purple = ["purple", [128,0,128]]
    brown = ["brown", [0, 75, 150]]
    orange = ["orange", [0,165,255]]
    colors = [yellow, white, black, red, blue, green, purple, brown, orange]
    
    # shapes
    shapes = ["triangle","rectangle","pentagon", "star"] # TODO: "circle", "semicircle", "quarter circle"
    
    size = 100  # size of the shape
    background_path = "./src/training/resources/backgrounds/pavement3.jpg"
    img = cv2.imread(background_path)
    
    # initialize directory
    os.chdir("./training_datasets/train/images")
    
    for color in colors:
        for shape in shapes:
            coord = (random.randint(0, np.size(img, 1)), random.randint(0, np.size(img, 0)))
            print("center of the shape: ", coord)
            img_with_shape = draw_object(img, shape, coord, size, color[1], "A")  

    cv2.imshow(img,img_with_shape)    
    cv2.waitKey(0)

if __name__ == "__main__":
    main()