from .drawshape import *
from .drawtext import *


def draw_object(bg_img, shape, coord, size, color, text):
    """
    Place object (shape+text) on the image
    """
    img, text_bbox = draw_rectangle(bg_img, coord, size, color)
    img = putText(img, text, text_bbox, size)
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    return img
    