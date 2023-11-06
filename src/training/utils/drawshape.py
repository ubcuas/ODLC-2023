import cv2
import numpy as np
import largestinteriorrectangle as lir


def find_bbox_from_vertices(img, vertices):
    x1 = img.shape[1]
    y1 = img.shape[0]
    x2 = 0
    y2 = 0
    vertices = vertices[0]
    for vertex in vertices:
        x1 = min(vertex[0], x1)
        y1 = min(vertex[1], y1)
        x2 = max(vertex[0], x2)
        y2 = max(vertex[1], y2)

    if y1 < 0:
        y1 = 0
    if y2 > img.shape[0]:
        y2 = img.shape[0]
    if x1 < 0:
        x1 = 0
    if x2 > img.shape[1]:
        x2 = img.shape[1]
    return ((x1, y1), (x2, y2))


def draw_circle(img, coord, size, color):
    radius = int(size / 2)
    side_length = int(np.sqrt(np.square(radius) / 2))
    text_bbox = (
        (coord[0] - side_length, coord[1] - side_length),
        (coord[0] + side_length, coord[1] + side_length),
    )
    object_bbox = (
        (coord[0] - int(size / 2), coord[1] - int(size / 2)),
        (coord[0] + int(size / 2), coord[1] + int(size / 2)),
    )
    img = cv2.circle(img, coord, radius, color, cv2.FILLED)
    return img, text_bbox, object_bbox


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
    object_bbox = (
        (coord[0] - int(size / 2), coord[1] - int(size / 4)),
        (coord[0] + int(size / 2), coord[1] + int(size / 4)),
    )
    img = cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, cv2.FILLED)
    return img, text_bbox, object_bbox


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
    object_bbox = (
        (coord[0] - int(size / 2), coord[1] - int(size / 2)),
        (coord[0] + int(size / 2), coord[1] + int(size / 2)),
    )
    return img, text_bbox, object_bbox


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
    object_bbox = (
        (coord[0] - int(size / 2), coord[1] - int(size / 2)),
        (coord[0] + int(size / 2), coord[1] + int(size / 2)),
    )
    return img, text_bbox, object_bbox


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


def draw_pentagon(img, coord, size, color):
    # Calculate the coordinates of the pentagon vertices
    vertices = np.array(
        [
            [
                [
                    ((size / 2) * np.cos(np.radians(i))) + coord[0],
                    ((size / 2) * np.sin(np.radians(i))) + coord[1],
                ]
                for i in range(342, 0, -72)
            ]
        ],
        np.int32,
    )
    text_bbox = [
        (
            vertices[0][3][0],
            vertices[0][3][1] - (vertices[0][4][0] - vertices[0][3][0]),
        ),
        vertices[0][4],
    ]
    object_bbox = find_bbox_from_vertices(img, vertices)
    # Draw the pentagon on the image
    img = cv2.fillPoly(img, vertices, color=color)
    return img, text_bbox, object_bbox


# TODO: test cases that biggest rectangle is a tall rectangle (if it even possible?), adjust inner diameter of a star
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
    vertices = np.array([vertices], np.int32)
    text_bbox = lir.lir(vertices)
    if text_bbox[2] > text_bbox[3]:
        offset = int((text_bbox[2] - text_bbox[3]) / 2)
        text_bbox = (
            (text_bbox[0] + offset, text_bbox[1]),
            (text_bbox[0] + offset + text_bbox[3], text_bbox[1] + text_bbox[3]),
        )
    object_bbox = find_bbox_from_vertices(img, vertices)
    # Draw the star on the image
    img = cv2.fillPoly(img, vertices, color=color)
    return img, text_bbox, object_bbox


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
    object_bbox = (
        (coord[0] - int(size / 2), coord[1] - int(size / 2)),
        (coord[0] + int(size / 2), coord[1] + int(size / 2)),
    )
    return img, text_bbox, object_bbox


if __name__ == "__main__":
    import copy
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="output/")
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    bg_img = np.ones((600, 800, 3), np.uint8) * 255

    def visualise(img, text_bbox, object_bbox):
        img = cv2.rectangle(img, text_bbox[0], text_bbox[1], (255, 255, 0), 3)
        img = cv2.rectangle(img, object_bbox[0], object_bbox[1], (0, 0, 0), 3)
        return img

    # circle
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_circle(img, (200, 200), 100, (0, 0, 255))
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("circle.jpg")), img)

    # semicircle
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_semicircle(
        img, (200, 200), 100, (0, 0, 255), direction=0
    )
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("semicircle.jpg")), img)

    # quater circle
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_quarter_circle(
        img, (200, 200), 100, (0, 0, 255), direction=0
    )
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("quater_circle.jpg")), img)

    # triangle
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_triangle(img, (200, 200), 100, (0, 0, 255))
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("triangle.jpg")), img)

    # rectangle
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_rectangle(img, (200, 200), 100, (0, 0, 255))
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("rectangle.jpg")), img)

    # cross
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_cross(img, (200, 200), 100, (0, 0, 255), cross_corner=0.5)
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("cross.jpg")), img)

    # pentagon
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_pentagon(img, (200, 200), 200, (0, 0, 255))
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("pentagon.jpg")), img)

    # star
    img = copy.deepcopy(bg_img)
    img, text_bbox, object_bbox = draw_star(img, (200, 200), 300, (0, 0, 255))
    img = visualise(img, text_bbox, object_bbox)
    cv2.imwrite(str(output_path.joinpath("star.jpg")), img)
