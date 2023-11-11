import cv2
import os
import numpy as np
from utils.commonDataStructure import colors_dictionary, shapes_dictionary


def saveImageAndLabel(img, shape, color, bounding_box, idx):
    """
    create training data image
    """
    # write image to the file
    os.chdir("../images")  # change to the right directory
    image_file = str(color[0]) + "_" + str(shape) + "_" + str(idx) + ".jpg"     # create image file
    cv2.imwrite(image_file, img)

    # create corresponding label file
    os.chdir("../labels")
    addLabel(bounding_box, color[0], shape, idx, np.size(img, 0), np.size(img, 1))
    
    return img


def addLabel(bounding_box, color, shape, idx, img_height, img_width):
    # add labels
    label = open(str(color) + "_" + str(shape) + "_" + str(idx) + ".txt", "w")
    height = (bounding_box[1][1] - bounding_box[0][1])/img_height
    width = (bounding_box[1][0] - bounding_box[0][0])/img_width
    y_center = ((bounding_box[1][1] + bounding_box[0][1])/2)/img_height
    x_center = ((bounding_box[1][0] + bounding_box[0][0])/2)/img_width
    label.write(str(colors_dictionary[color]) + str(shapes_dictionary[shape]) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height))
    label.close
    
