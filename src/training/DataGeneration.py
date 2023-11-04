import cv2
import numpy as np
from utils.drawobject import draw_object
from utils.saveImgLabel import saveImageAndLabel
from utils.imageAug import imageAugmentation
import random
import os
import copy
    
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
    # colors = [yellow, white, black, red, blue, green, purple, brown, orange]
    colors = [yellow, red]
    
    # shapes
    shapes = ["triangle", "rectangle"] # TODO: "pentagon", "star", "circle", "semicircle", "quarter circle"
    
    size = 100  # size of the shape
    background_path = "./src/training/resources/backgrounds/pavement3.jpg"
    bg_img = cv2.imread(background_path)
    
    # initialize directory
    os.chdir("./src/training/training_datasets/train/images")
    
    for color in colors:
        for shape in shapes:
            img = copy.deepcopy(bg_img)
            coord = (random.randint(0, np.size(img, 1)), random.randint(0, np.size(img, 0)))
            print("center of the shape: ", coord)
            img_with_shape, bounding_box = draw_object(img, shape, coord, size, color[1], "orange", "A")  
            aug_imgs, new_bounding_boxes = imageAugmentation(img_with_shape, bounding_box)
            for i in range(len(aug_imgs)):
                print("i: ", i)
                saveImageAndLabel(aug_imgs[i], shape, color, new_bounding_boxes[i])

    # cv2.imshow(img,img_with_shape)    
    cv2.waitKey(0)

if __name__ == "__main__":
    dataGeneration()
    
print("end of DataGeneration")