import cv2
import numpy as np
from utils.drawobject import draw_object
from utils.saveImgLabel import saveImageAndLabel
from utils.imageAug import imageAugmentation
import utils.commonDataStructure as cds
from utils.commonDataStructure import ObjectLabel
import random
import os
import copy
from pathlib import Path
import string
import time


def check_overlap(coord: tuple[int, int], placed_objs: list[ObjectLabel]):
    if placed_objs:
        for obj in placed_objs:
            if obj.pt1[0] < coord[0] < obj.pt2[0]:
                return True
            if obj.pt1[1] < coord[1] < obj.pt2[1]:
                return True
    return False


def dataGeneration(bg_path: Path, output_path: Path, num_gen: int):
    """Creates images with different shapes, colors and letters on it"""

    # settings
    size_min = 30  # size of the shape
    size_max = 55
    background_paths = list(bg_path.glob("*"))
    num_augimg_per_original = 3
    max_obj_per_img = 20
    min_obj_per_img = 15

    # initialize directory
    output_path.joinpath("images").mkdir(exist_ok=True, parents=True)
    output_path.joinpath("labels").mkdir(exist_ok=True, parents=True)

    characters = string.ascii_uppercase
    idx = 0
    generated_objs = {}
    
    # iterating over every background to minimize reading from disk
    for bg in background_paths:
        bg_img = cv2.imread(str(bg))
        bg_img_height, bg_img_width, _ = bg_img.shape
        # use all the background images equally
        for _ in range(int(num_gen / len(background_paths))):
            # pick number of object
            num_obj = random.randint(min_obj_per_img, max_obj_per_img)
            img = copy.deepcopy(bg_img)
            size = random.randint(size_min, size_max)
            # place each object onto the image
            placed_objs = []
            for _ in range(num_obj):
                shape = random.choice(cds.shapes)
                shape_color = random.choice(cds.colors)
                text_color = random.choice(cds.colors)
                while text_color == shape_color:
                    text_color = random.choice(cds.colors)
                char = random.choice(characters)
                coord = (
                    random.randint(int(size / 2), np.size(img, 1) - int(size / 2)),
                    random.randint(int(size / 2), np.size(img, 0) - int(size / 2)),
                )
                if check_overlap(coord, placed_objs):
                    print("overlapped shapes")
                    continue
                img, bbox = draw_object(
                    img, shape, coord, size, shape_color[1], text_color[1], char
                )
                obj = ObjectLabel(
                    bbox[0],
                    bbox[1],
                    bg_img_width,
                    bg_img_height,
                    shape,
                    shape_color,
                    text_color,
                )
                placed_objs.append(obj)
                if obj.name in generated_objs:
                    generated_objs[obj.name] += 1
                else:
                    generated_objs[obj.name] = 1
            
            # this part is modified so that the images are augmented
            # ================================================
            bounding_boxes = []
            
            # get the bounding boxes of the shapes
            for object in placed_objs:
                bounding_boxes.append([object.pt1, object.pt2])
                
            # augmentation
            aug_imgs, new_bounding_boxes = imageAugmentation(
                img,
                bounding_boxes,
                bg_img_height,
                bg_img_width,
                num_aug_imgs=num_augimg_per_original,
            )
            
            # save images
            for i in range(len(aug_imgs)):
                saveImageAndLabel(
                    aug_imgs[i],
                    output_path,
                    [ObjectLabel(
                        new_bounding_boxes[i][j][0],
                        new_bounding_boxes[i][j][1],
                        bg_img_width,
                        bg_img_height,
                        placed_objs[j].shape,
                        placed_objs[j].shape_color,
                        "",
                    ) for j in range(len(new_bounding_boxes[i]))],
                    idx,
                )
                idx += 1
            # =============================================

    print(generated_objs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    output_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        # "./training_datasets/train"
        "./training_datasets/valid"
        # "../predict/images1"
    )
    default_amount = 10
    parser.add_argument("-o", "--output", default=output_path)
    input_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "./resources/backgrounds"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=input_path,
        help="Backgroud image folder to put object on",
    )
    parser.add_argument(
        "-a",
        "--amount",
        type=int,
        default=default_amount,
        help="Amount of image to generate",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    input_path = Path(args.input)
    if not input_path.exists():
        raise Exception("Input a valid input path")
    start_t = time.time()
    dataGeneration(input_path, output_path, args.amount)
    print(f"Total time to generate images: {time.time() - start_t}")


print("end of DataGeneration")