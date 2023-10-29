import cv2
import numpy as np
import copy
import time
from utils.drawshape import *

ft = cv2.freetype.createFreeType2()


def putText(img, text, text_bbox, font_size: float = 1.0):
    """
    Args:
        font_size: percentage of font size
    """
    ft.loadFontData(fontFileName="src/model/resources/fonts/arial.ttf", idx=0)
    font_height = int((text_bbox[1][1] - text_bbox[0][1]) * font_size)
    width, height = ft.getTextSize(text, font_height, -1)[0]
    print(height)
    ft.putText(
        img=img,
        text=text,
        org=(
            text_bbox[0][0] + int(((text_bbox[1][0] - text_bbox[0][0]) - width) / 2),
            text_bbox[1][1] - int(((text_bbox[1][1] - text_bbox[0][1]) - height) / 2),
        ),
        fontHeight=font_height,
        color=(255, 255, 255),
        thickness=-1,
        line_type=cv2.LINE_AA,
        bottomLeftOrigin=True,
    )

    return img


def createImage(bg_img, shape, coord, size, color, text):
    """
    create training data image
    """
    img = copy.deepcopy(bg_img)
    img, text_bbox = draw_rectangle(img, coord, size, color)
    img = putText(img, text, text_bbox, size)
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    return img


size = 100
coord = (300, 300)
color = (255, 0, 0)
text = "W"
bg_path = "src/model/resources/backgrounds/pavement3.jpg"
bg_img = cv2.imread(bg_path)
img = copy.deepcopy(bg_img)
# bg_img = np.ones((1080, 1920, 3), np.uint8) * 255
t = time.time()
# shape_list = ["circle", "semicircle", "quarter circle", "triangle", "rectangle", "pentagon", "star", "cross"]
# for shape in shape_list:
#     img = createImage(bg_img, shape, coord, size, color, "A")
#     cv2.imwrite(f"output/{shape}.jpg", img)
# for i in range(1000):
#     img = createImage(bg_img, "quarter circle", coord, size, color, "A")

img, text_bbox = draw_cross(img, coord, size, color)
print(int((text_bbox[1][1] - text_bbox[0][1]) * 1))
img = putText(img, text, text_bbox, 1)
print(time.time() - t)

cv2.imshow("img", img)
cv2.waitKey(0)
