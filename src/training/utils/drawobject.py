from .drawshape import (
    draw_circle,
    draw_semicircle,
    draw_quarter_circle,
    draw_triangle,
    draw_rectangle,
    draw_pentagon,
    draw_star,
    draw_cross,
)
from .drawtext import draw_text
import random
import numpy as np


def draw_object(
    bg_img: np.ndarray[np.uint8],
    shape: str,
    coord: tuple[int, int],
    size: int,
    shape_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
    text: str,
    font_size: float = 1.0,
):
    """
    Place object (shape+text) on the image

    Args:
        bg_img: np array / opencv image to draw object on
        shape: can be one of: ["circle", "semicircle", "quarter circle", "triangle", "rectangle", "pentagon", "star", "cross"]
        coord: center (x,y) coordinate of the object
        size: maximum possible width and higth of the object
        shape_color: (b,g,r) color of the shape
        text_color: ((b,g,r) color of the text
        text: character to draw
        font_size: font size in percentage of maximum possible font size to put in an object (default to 1.0)
    """
    if shape == "circle":
        img, text_bbox, object_bbox = draw_circle(bg_img, coord, size, shape_color)
    elif shape == "semicircle":
        img, text_bbox, object_bbox = draw_semicircle(bg_img, coord, size, shape_color)
    elif shape == "quarter circle":
        img, text_bbox, object_bbox = draw_quarter_circle(
            bg_img, coord, size, shape_color
        )
    elif shape == "triangle":
        img, text_bbox, object_bbox = draw_triangle(bg_img, coord, size, shape_color)
    elif shape == "rectangle":
        img, text_bbox, object_bbox = draw_rectangle(bg_img, coord, size, shape_color)
    elif shape == "pentagon":
        img, text_bbox, object_bbox = draw_pentagon(bg_img, coord, size, shape_color)
    elif shape == "star":
        img, text_bbox, object_bbox = draw_star(bg_img, coord, size, shape_color)
    elif shape == "cross":
        cross_corner = random.uniform(0.3, 0.7)
        img, text_bbox, object_bbox = draw_cross(
            bg_img, coord, size, shape_color, cross_corner
        )
    img = draw_text(img, text, text_bbox, text_color, font_size)
    return img, object_bbox
