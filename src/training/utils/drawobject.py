from .drawshape import *
from .drawtext import *


def draw_object(bg_img, shape, coord, size, shape_color, text_color, text, font_size:float = 1.0):
    """
    Place object (shape+text) on the image
    """
    img, text_bbox, object_bbox = draw_rectangle(bg_img, coord, size, shape_color)
    img = putText(img, text, text_bbox, text_color, font_size)
    return img, object_bbox
    