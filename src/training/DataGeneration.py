import cv2
import numpy as np
from utils.drawobject import draw_object
from utils.saveImgLabel import saveImageAndLabel
from utils.imageAug import imageAugmentation
import utils.commonDataStructure as cds
import random
import os
import copy
from pathlib import Path
import string
import time


def dataGeneration(bg_path: Path, output_path: Path):
    # creates images with different shapes and colors

    size = 100  # size of the shape
    background_paths = list(bg_path.glob("*"))

    # initialize directory
    output_path.joinpath("images").mkdir(exist_ok=True, parents=True)
    output_path.joinpath("labels").mkdir(exist_ok=True, parents=True)

    idx = 0
    total_img = 0
    # characters = string.ascii_uppercase
    characters = ["A"]
    print(f"Number of Image being generated: {len(background_paths) * len(cds.colors) * (len(cds.colors) - 1) * 1 * 10}")
    for bg in background_paths:
        print(str(bg))
        # bg_img = cv2.imread(str(bg))
        bg_img = cv2.imread("src/training/resources/backgrounds/pavement.jpg")
        print(bg_img.shape)
        for color in cds.colors:
            for shape in cds.shapes:
                idx = 0
                for text_color in cds.colors:
                    if color == text_color:
                        continue
                    for char in characters:
                        img = copy.deepcopy(bg_img)
                        coord = (
                            random.randint(int(size / 2), np.size(img, 1) - int(size / 2)),
                            random.randint(int(size / 2), np.size(img, 0) - int(size / 2)),
                        )
                        print("center of the shape: ", coord)
                        img_with_shape, bounding_box = draw_object(
                            img, shape, coord, size, color[1], text_color[1], char
                        )
                        aug_imgs, new_bounding_boxes = imageAugmentation(
                            img_with_shape, bounding_box
                        )
                        print(total_img)
                        for i in range(len(aug_imgs)):
                            saveImageAndLabel(
                                aug_imgs[i], output_path, shape, color, new_bounding_boxes[i], idx
                            )
                            total_img += 1
                            idx += 1

    # cv2.imshow(img,img_with_shape)
    # cv2.waitKey(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="output/")
    parser.add_argument(
        "-i",
        "--input",
        default="backgroud_images/",
        help="Backgroud image folder to put object on",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    input_path = Path(args.input)
    if not input_path.exists():
        raise Exception("Input a valid input path")
    start_t = time.time()
    dataGeneration(input_path, output_path)
    print(f"Total time to generate images: {time.time() - start_t}")

print("end of DataGeneration")
