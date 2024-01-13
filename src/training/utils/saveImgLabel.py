import cv2
import os
import numpy as np
import utils.commonDataStructure as cds
from utils.commonDataStructure import ObjectLabel
from pathlib import Path


def saveImageAndLabel(img, output_path: Path, placed_objs: list[ObjectLabel], idx):
    """
    create training data image
    """
    # write image to the file
    image_file = output_path.joinpath("images", f"{idx}.jpg")  # create image file
    cv2.imwrite(str(image_file), img)

    # create corresponding label file
    addLabel(output_path, placed_objs, idx)

    return img


def addLabel(output_path: Path, placed_objs: list[ObjectLabel], idx):
    # add labels
    label = open(output_path.joinpath("labels", f"{idx}.txt"), "w")
    for obj in placed_objs:
        name_color_shape = obj.shape_color[0] + "_" + obj.shape
        label.write(
            str(cds.lookup_table.index(name_color_shape))
            + " "
            + str(obj.get_normalized_centerx())
            + " "
            + str(obj.get_normalized_centery())
            + " "
            + str(obj.get_normalized_width())
            + " "
            + str(obj.get_normalized_height())
            + "\n"
        )
    label.close
    