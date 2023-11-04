import cv2
import numpy as np


def draw_circle(img, coord, size, color):
    radius = int(size / 2)
    side_length = int(np.sqrt(np.square(radius) / 2))
    text_bbox = (
        (coord[0] - side_length, coord[1] - side_length),
        (coord[0] + side_length, coord[1] + side_length),
    )
    img = cv2.circle(img, coord, radius, color, cv2.FILLED)
    return img, text_bbox


# TODO: implement direction
def draw_semicircle(img, coord, size, color, direction=0):
    radius = int(size / 2)
    axes = (radius, radius)
    angle = 0
    startAngle = 180
    endAngle = 360
    center = (coord[0], int(coord[1] + (size / 4)))
    side_length = int(np.sqrt((4 * np.square(radius)) / 5))
    text_bbox = (
        (coord[0] - int(side_length / 2), coord[1] + int(size / 4) - side_length),
        (coord[0] + int(side_length / 2), coord[1] + int(size / 4)),
    )
    img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, cv2.FILLED)
    return img, text_bbox


# TODO: implement direction
def draw_quarter_circle(img, coord, size, color, direction=0):
    radius = size
    axes = (radius, radius)
    angle = 0
    startAngle = 180
    endAngle = 270
    center = (int(coord[0] + (size / 2)), int(coord[1] + (size / 2)))
    cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, cv2.FILLED)
    side_length = int(np.sqrt(np.square(radius) / 2))
    text_bbox = (
        (
            (coord[0] + int(radius / 2)) - side_length,
            (coord[1] + int(radius / 2)) - side_length,
        ),
        (coord[0] + int(radius / 2), coord[1] + int(radius / 2)),
    )
    return img, text_bbox


# TODO: add option to change base - height ratio, fix text_bbox
def draw_triangle(img, coord, size, color):
    triangle_pts = np.array(
        [
            (int(coord[0] - (size / 2)), int(coord[1] + (size / 2))),
            (coord[0], int(coord[1] - (size / 2))),
            (int(coord[0] + (size / 2)), int(coord[1] + (size / 2))),
        ],
        np.int32,
    )
    img = cv2.fillPoly(img, [triangle_pts], color)
    side_length = np.square(size) / (size + size)
    text_bbox = (
        (coord[0] - int(side_length / 2), int(coord[1] + (size / 2) - side_length)),
        (coord[0] + int(side_length / 2), coord[1] + int(size / 2)),
    )
    return img, text_bbox


def draw_rectangle(img, coord, size, color):
    text_bbox = (
        (coord[0] - int(size / 2), coord[1] - int(size / 2)),
        (coord[0] + int(size / 2), coord[1] + int(size / 2)),
    )
    img = cv2.rectangle(
        img,
        (int(coord[0] - (size / 2)), int(coord[1] - (size / 2))),
        (int(coord[0] + (size / 2)), int(coord[1] + (size / 2))),
        color,
        cv2.FILLED,
    )
    return img, text_bbox, text_bbox


# TODO: create text_bbox
def draw_pentagon(img, coord, size, color):
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
    img = cv2.fillPoly(img, [vertices], color=color)
    return img


# TODO: create text_bbox
def draw_star(img, coord, size, color):
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
    img = cv2.fillPoly(img, [vertices], color=color)
    return img


def draw_cross(img, coord, size, color, cross_corner=0.5):
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
        ],
        np.int32,
    )
    img = cv2.fillPoly(img, [vertices], color=color)
    text_bbox = (
        (
            int(coord[0] - (size / 2) * cross_corner),
            int(coord[1] - (size / 2) * cross_corner),
        ),
        (
            int(coord[0] + (size / 2) * cross_corner),
            int(coord[1] + (size / 2) * cross_corner),
        ),
    )
    return img, text_bbox


if __name__ == "__main__":
    import copy

    bg_img = np.ones((600, 800, 3), np.uint8) * 255

    # circle
    img = copy.deepcopy(bg_img)
    img, text_bbox = draw_circle(img, (200, 200), 100, (0, 0, 255))
    img = cv2.rectangle(img, text_bbox[0], text_bbox[1], (255, 255, 0), 3)
    cv2.imwrite("output/shape/circle.jpg", img)

    # semicircle
    img = copy.deepcopy(bg_img)
    img, text_bbox = draw_semicircle(img, (200, 200), 100, (0, 0, 255), direction=0)
    img = cv2.rectangle(img, text_bbox[0], text_bbox[1], (255, 255, 0), 3)
    cv2.imwrite("output/shape/semicircle.jpg", img)

    # quater circle
    img = copy.deepcopy(bg_img)
    img, text_bbox = draw_quarter_circle(img, (200, 200), 100, (0, 0, 255), direction=0)
    img = cv2.rectangle(img, text_bbox[0], text_bbox[1], (255, 255, 0), 3)
    cv2.imwrite("output/shape/quater_circle.jpg", img)

    # triangle
    img = copy.deepcopy(bg_img)
    img, text_bbox = draw_triangle(img, (200, 200), 100, (0, 0, 255))
    img = cv2.rectangle(img, text_bbox[0], text_bbox[1], (255, 255, 0), 3)
    cv2.imwrite("output/shape/triangle.jpg", img)

    # rectangle
    img = copy.deepcopy(bg_img)
    img, text_bbox = draw_rectangle(img, (200, 200), 100, (0, 0, 255))
    img = cv2.rectangle(img, text_bbox[0], text_bbox[1], (255, 255, 0), 3)
    cv2.imwrite("output/shape/rectangle.jpg", img)

    # cross
    img = copy.deepcopy(bg_img)
    img, text_bbox = draw_cross(img, (200, 200), 100, (0, 0, 255))
    img = cv2.rectangle(img, text_bbox[0], text_bbox[1], (255, 255, 0), 3)
    cv2.imwrite("output/shape/cross.jpg", img)
