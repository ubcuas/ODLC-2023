import cv2
import os
import numpy as np
import utils.commonDataStructure as cds
from pathlib import Path


def saveImageAndLabel(img, output_path: Path, shape, color, bounding_box, idx):
    """
    create training data image
    """
    # write image to the file
    image_file = output_path.joinpath("images", str(color[0]) + "_" + str(shape) + "_" + str(idx) + ".jpg")     # create image file
    cv2.imwrite(str(image_file), img)

    # create corresponding label file
    addLabel(bounding_box, color[0], output_path, shape, idx, np.size(img, 0), np.size(img, 1))
    
    return img


def addLabel(bounding_box, color, output_path: Path, shape, idx, img_height, img_width):
    # add labels
    label = open(output_path.joinpath("labels", str(color) + "_" + str(shape) + "_" + str(idx) + ".txt"), "w")
    height = (bounding_box[1][1] - bounding_box[0][1])/img_height
    width = (bounding_box[1][0] - bounding_box[0][0])/img_width
    y_center = ((bounding_box[1][1] + bounding_box[0][1])/2)/img_height
    x_center = ((bounding_box[1][0] + bounding_box[0][0])/2)/img_width
    name_color_shape = color + "_" + shape
    label.write(str(cds.lookup_table.index(name_color_shape)) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height))
    label.close
    
