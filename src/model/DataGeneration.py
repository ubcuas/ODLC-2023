import cv2
import numpy as np
import copy
import time


def placeShape(img, shape, coord, size, color, cross_corner = 0.5):
    fontHeight = 0
    if shape == "circle":
        fontHeight = size
        return cv2.circle(img, coord, int(size / 2), color, cv2.FILLED)
    elif shape == "semicircle":
        radius = int(size / 2)
        fontHeight = radius
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
        fontHeight = size
        axes = (radius, radius)
        angle = 0
        startAngle = 180
        endAngle = 270
        center = (int(coord[0] + (size / 2)), int(coord[1] + (size / 2)))
        return cv2.ellipse(
            img, center, axes, angle, startAngle, endAngle, color, cv2.FILLED
        )
    elif shape == "triangle":
        fontHeight = size
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
        fontHeight = size
        return cv2.rectangle(
            img,
            (int(coord[0] - (size / 2)), int(coord[1] - (size / 2))),
            (int(coord[0] + (size / 2)), int(coord[1] + (size / 2))),
            color,
            cv2.FILLED,
        )
    elif shape == "pentagon":
        fontHeight = size
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
        fontHeight = size
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
        fontHeight = size
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


def putText(img, text, coord, fontHeight):
    ft = cv2.freetype.createFreeType2()

    ft.loadFontData(fontFileName='src/model/resources/fonts/arial.ttf',
                idx=0)
    ft.putText(img=img,
            text=text,
            org=(coord[0], coord[1] - int(fontHeight/2)),
            fontHeight=fontHeight,
            color=(255, 255, 255),
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=False)

    return img


def createImage(bg_img, shape, coord, size, color, text):
    """
    create training data image
    """
    img = copy.deepcopy(bg_img)
    img = placeShape(img, shape, coord, size, color)
    img = putText(img, text, coord, size)
    # img = cv2.GaussianBlur(img, (7, 7), 0)
    return img


size = 100
coord = (300, 150)
color = (255, 0, 0)
bg_path = "src/model/resources/backgrounds/pavement3.jpg"
bg_img = cv2.imread(bg_path)
t = time.time()
shape_list = ["circle", "semicircle", "quarter circle", "triangle", "rectangle", "pentagon", "star", "cross"]
for shape in shape_list:
    img = createImage(bg_img, shape, coord, size, color, "A")
    cv2.imwrite(f"output/{shape}.jpg", img)
print(time.time() - t)
# cv2.imshow("img", img)
# cv2.waitKey(0)
