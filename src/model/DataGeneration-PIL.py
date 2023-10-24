import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os


def placeShape(img, shape, coord, size, color, cross_corner = 0.5):
    if shape == "circle":
        return cv2.circle(img, coord, int(size / 2), color, cv2.FILLED)
    elif shape == "semicircle":
        radius = int(size / 2)
        axes = (radius, radius)
        angle = 0
        startAngle = 180
        endAngle = 360
        center = (coord[0], int(coord[1] + (size / 4)))
        return cv2.ellipse(
            img, center, axes, angle, startAngle, endAngle, color, cv2.FILLED
        )
    elif shape == "quarter circle":
        radius = size
        axes = (radius, radius)
        angle = 0
        startAngle = 180
        endAngle = 270
        center = (int(coord[0] + (size / 2)), int(coord[1] + (size / 2)))
        return cv2.ellipse(
            img, center, axes, angle, startAngle, endAngle, color, cv2.FILLED
        )
    elif shape == "triangle":
        triangle_pts = np.array(
            [
                (int(coord[0] - (size / 2)), int(coord[1] + (size / 2))),
                (coord[0], int(coord[1] - (size / 2))),
                (int(coord[0] + (size / 2)), int(coord[1] + (size / 2))),
            ],
            np.int32,
        )
        return cv2.fillPoly(img, [triangle_pts], color)
    elif shape == "rectangle":
        return cv2.rectangle(
            img,
            (int(coord[0] - (size / 2)), int(coord[1] - (size / 2))),
            (int(coord[0] + (size / 2)), int(coord[1] + (size / 2))),
            color,
            cv2.FILLED,
        )
    elif shape == "pentagon":
        # Calculate the coordinates of the pentagon vertices
        vertices = np.array(
            [
                [
                    ((size / 2) * np.cos(np.radians(i))) + coord[0],
                    ((size / 2) * np.sin(np.radians(i))) + coord[1],
                ]
                for i in range(342, 0, -72)
            ],
            np.int32,
        )

        # Draw the pentagon on the image
        return cv2.fillPoly(img, [vertices], color=color)
    elif shape == "star":
        outter_vertices = np.array(
            [
                [
                    ((size / 2) * np.cos(np.radians(i))) + coord[0],
                    ((size / 2) * np.sin(np.radians(i))) + coord[1],
                ]
                for i in range(342, 0, -72)
            ],
            np.int32,
        )
        inner_vertices = np.array(
            [
                [
                    ((size / 6) * np.cos(np.radians(i))) + coord[0],
                    ((size / 6) * np.sin(np.radians(i))) + coord[1],
                ]
                for i in range(18, 360, 72)
            ],
            np.int32,
        )[::-1]
        vertices = np.empty((10, 2), np.int32)
        vertices[0::2] = outter_vertices
        vertices[1::2] = inner_vertices

        # Draw the star on the image
        return cv2.fillPoly(img, [vertices], color=color)
    elif shape == "cross":
        vertices = np.array(
            [
                (coord[0] - (size / 2) * cross_corner, coord[1] - size / 2),
                (coord[0] + (size / 2) * cross_corner, coord[1] - size / 2),
                (
                    coord[0] + (size / 2) * cross_corner,
                    coord[1] - (size / 2) * cross_corner,
                ),
                (coord[0] + size / 2, coord[1] - (size / 2) * cross_corner),
                (coord[0] + size / 2, coord[1] + (size / 2) * cross_corner),
                (
                    coord[0] + (size / 2) * cross_corner,
                    coord[1] + (size / 2) * cross_corner,
                ),
                (coord[0] + (size / 2) * cross_corner, coord[1] + size / 2),
                (coord[0] - (size / 2) * cross_corner, coord[1] + size / 2),
                (
                    coord[0] - (size / 2) * cross_corner,
                    coord[1] + (size / 2) * cross_corner,
                ),
                (coord[0] - size / 2, coord[1] + (size / 2) * cross_corner),
                (coord[0] - size / 2, coord[1] - (size / 2) * cross_corner),
                (
                    coord[0] - (size / 2) * cross_corner,
                    coord[1] - (size / 2) * cross_corner,
                ),
            ], np.int32
        )
        return cv2.fillPoly(img, [vertices], color=color)


def putText(img, text, coord, size):
    # Convert the image to RGB (OpenCV uses BGR)
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pass the image to PIL
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # use a truetype font
    # font = ImageFont.truetype("Tahoma Regular font.ttf", 100)
    font = ImageFont.truetype("src/model/resources/fonts/arial.ttf", 75)
    # font = ImageFont.load_default()

    # Draw the text
    draw.text(
        # (coord[0] - (size / 2), coord[1] - (size / 2)),
        coord,
        text,
        font=font,
        fill=(255, 255, 255, 255),
        anchor="mm",
    )

    # Get back the image to OpenCV
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def createImage(bg_path, shape, coord, size, color, text):
    """
    create training data image
    """
    img = cv2.imread(bg_path)
    img = placeShape(img, shape, coord, size, color)
    img = putText(img, text, coord, size)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    return img


size = 100
coord = (300, 150)
color = (255, 0, 0)
bg_path = "src/model/resources/backgrounds/pavement3.jpg"
img = createImage(bg_path, "cross", coord, size, color, "A")

cv2.imshow("img", img)
cv2.waitKey(0)
