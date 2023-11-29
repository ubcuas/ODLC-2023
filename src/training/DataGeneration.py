import cv2
import numpy as np
from utils.drawobject import draw_object
from utils.saveImgLabel import saveImageAndLabel
from utils.imageAug import imageAugmentation
import utils.commonDataStructure as cds
import random
import math
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
    num_augimg_per_original = 10
    characters = cds.characters
    print(
        f"Number of Image being generated: {len(background_paths) * len(cds.shapes) * len(cds.colors) * (len(cds.colors) - 1) * (len(characters)) * num_augimg_per_original}"
    )
    
    for color in cds.colors:
        for shape in cds.shapes:
            idx = 0
            for bg in background_paths:
                bg_img = cv2.imread(str(bg))
                for text_color in cds.colors:
                    if color == text_color:
                        continue
                    for char in characters:
                        img = copy.deepcopy(bg_img)
                        
                        # # old coord generation code
                        # coord = (
                        #     random.randint(
                        #         int(size / 2), np.size(img, 1) - int(size / 2)
                        #     ),
                        #     random.randint(
                        #         int(size / 2), np.size(img, 0) - int(size / 2)
                        #     ),
                        # )

                        # generates coord so that it lies within a circle centered at the middle of the image
                        # this makes it so that the shape won't go out-of-bounds if the image is rotated

                        # radius of the circle
                        if np.size(img, 1) >= np.size(img, 0):
                            circle_r = (np.size(img, 0) - size)/2
                        else:
                            circle_r = (np.size(img, 1) - size)/2
                        
                        # center of the circle (x, y)
                        circle_x = np.size(img, 1)/2
                        circle_y = np.size(img, 0)/2

                        # random angle
                        alpha = 2 * math.pi * random.random()

                        # random radius
                        r = circle_r * math.sqrt(random.random())

                        # calculating coordinates
                        x_coord = r * math.cos(alpha) + circle_x
                        y_coord = r * math.sin(alpha) + circle_y
                        coord = (int(x_coord), int(y_coord))

                        # print("center of the shape: ", coord)
                        img_with_shape, bounding_box = draw_object(
                            img, shape, coord, size, color[1], text_color[1], char
                        )
                        bg_img_height, bg_img_width, channels = bg_img.shape
                        aug_imgs, new_bounding_boxes = imageAugmentation(
                            img_with_shape, bounding_box, bg_img_height, bg_img_width,num_aug_imgs=num_augimg_per_original)
                        for i in range(len(aug_imgs)):
                            saveImageAndLabel(
                                aug_imgs[i],
                                output_path,
                                shape,
                                color,
                                new_bounding_boxes[i],
                                idx,
                            )
                            total_img += 1
                            idx += 1
                    print(total_img)

    # cv2.imshow(img,img_with_shape)
    # cv2.waitKey(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./training_datasets/valid")
    parser.add_argument("-o", "--output", default=output_path)
    input_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./resources/backgrounds")
    parser.add_argument(
        "-i",
        "--input",
        default=input_path,
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
