import cv2
import numpy as np
from utils.drawobject import draw_object
from utils.saveImgLabel import saveImageAndLabel
import random
import os
    
def dataGeneration():
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
    os.chdir("./src/training/training_datasets/train/images")
    
    for color in colors:
        for shape in shapes:
            coord = (random.randint(0, np.size(img, 1)), random.randint(0, np.size(img, 0)))
            print("center of the shape: ", coord)
            img_with_shape, bounding_box = draw_object(img, shape, coord, size, color[1], "A")  
            saveImageAndLabel(img_with_shape, shape, color, bounding_box)

    # cv2.imshow(img,img_with_shape)    
    cv2.waitKey(0)

if __name__ == "__main__":
    dataGeneration()
    
print("end of DataGeneration")