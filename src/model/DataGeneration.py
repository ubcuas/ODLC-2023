import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def create_img(img, shape, coord, size, color):
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
            background_img,
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




def putText(img, text, coord, size):
    # Convert the image to RGB (OpenCV uses BGR)
    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pass the image to PIL
    pil_im = Image.fromarray(cv2_im_rgb)

    draw = ImageDraw.Draw(pil_im)

    # use a truetype font
    # font = ImageFont.truetype("Tahoma Regular font.ttf", 100)
    font = ImageFont.truetype(".resources/fonts/arial.ttf", 75)
    # font = ImageFont.load_default()

    # Draw the text
    draw.text(
        # (coord[0] - (size / 2), coord[1] - (size / 2)),
        coord,
        text,
        font=font,
        fill=(255, 255, 255, 255),
        anchor="mm"
    )

    # Get back the image to OpenCV
    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

background_img = np.ones((600, 800, 3), np.uint8) * 255
size = 100
coord = (200, 300)
# background_img = cv2.rectangle(
#     background_img,
#     (int(coord[0] - (size / 2)), int(coord[1] - (size / 2))),
#     (int(coord[0] + (size / 2)), int(coord[1] + (size / 2))),
#     (0, 0, 0),
#     3,
# )
background_img = create_img(background_img, "pentagon", coord, size, (0, 0, 255))
background_img = putText(background_img, "C", coord, size)
# background_img = create_img(background_img, "semicircle", coord, size, (0, 0, 255))
# background_img = create_img(background_img, "circle", coord, size, (0, 0, 255))

cv2.imshow("img", background_img)
cv2.waitKey(0)