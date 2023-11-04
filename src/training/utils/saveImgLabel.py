import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os

def saveImageAndLabel(img, shape, color, bounding_box):
    """
    create training data image
    """
    # write image to the file
    os.chdir("../images")  # change to the right directory
    image_file = str(color[0]) + "_" + str(shape) + ".jpg"     # create image file
    cv2.imwrite(image_file, img)

    # create corresponding label file
    os.chdir("../labels")
    addLabel(bounding_box, color[0], shape)
    
    return img


def addLabel(bounding_box, color, shape):
    # add labels
    label = open(str(color) + "_" + str(shape) + ".txt", "w")
    height = bounding_box[1][1] - bounding_box[0][1]
    width = bounding_box[1][0] - bounding_box[0][0]
    y_center = (bounding_box[1][1] + bounding_box[0][1])/2
    x_center = (bounding_box[1][0] + bounding_box[0][0])/2
    label.write("0 " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height))
    label.close
    
