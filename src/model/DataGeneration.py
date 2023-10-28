import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import random


def placeShape(img_original, shape, coord, size, color):
    img = img_original.copy()
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
        vertices = np.array(
            [
                (int(coord[0] - (size / 2)), int(coord[1] + (size / 2))),
                (coord[0], int(coord[1] - (size / 2))),
                (int(coord[0] + (size / 2)), int(coord[1] + (size / 2))),
            ],
            np.int32,
        )
    elif shape == "rectangle":
        vertices = np.array(
            [
               [int(coord[0] - (size / 2)), int(coord[1] - (size / 2))],
               [int(coord[0] + (size / 2)), int(coord[1] - (size / 2))],
               [int(coord[0] + (size / 2)), int(coord[1] + (size / 2))],
               [int(coord[0] - (size / 2)), int(coord[1] + (size / 2))],
            ],
            np.int32,
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

    # find the boinding box
    [highest, lowest, leftmost, rightmost] = findBoundingBox(img, vertices, shape)
    
    # Draw the shape on the image
    return cv2.fillPoly(img, [vertices], color=color), [highest, lowest, leftmost, rightmost]


def findBoundingBox(img, vertices, shape):
    # find the highest, lowest, leftmost and rightmost points
    
    if(shape in ["circle", "semicircle", "quarter circle"]):
        pass # TODO
    else:    
        highest = img.shape[0]
        lowest = 0
        leftmost = img.shape[1]
        rightmost = 0
        for vertex in vertices:
            if vertex[0] <= leftmost:
                leftmost = vertex[0]
            if vertex[0] >= rightmost:
                rightmost = vertex[0]
            if vertex[1] <= highest:
                highest = vertex[1]
            if vertex[1] >= lowest:
                lowest = vertex[1]
            
    if(highest < 0):
        highest = 0
    if(lowest > img.shape[0]):
        lowest = img.shape[0]
    if(leftmost < 0 ):
        leftmost = 0
    if(rightmost > img.shape[1]):
        rightmost = img.shape[1]
        
    highest /= img.shape[0]
    lowest /= img.shape[0]
    leftmost /= img.shape[1]
    rightmost /= img.shape[1]
    print("the highest (y), lowest (y), leftmost (x) and rightmost (x) coords (normalized): ", highest, lowest, leftmost, rightmost)
    return [highest, lowest, leftmost, rightmost]

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

def createImage(img_original, shape, coord, size, color, text):
    """
    create training data image
    """
    img, [highest, lowest, leftmost, rightmost] = placeShape(img_original, shape, coord, size, color[1])
    img = putText(img, text, coord, size)
    img = cv2.GaussianBlur(img, (7,7), 0)

    # write image to the file
    os.chdir("../images")  # change to the right directory
    image_file = str(color[0]) + "_" + str(shape) + ".jpg"     # create image file
    cv2.imwrite(image_file, img)

    # create corresponding label file
    os.chdir("../labels")
    addLabel(highest, lowest, leftmost, rightmost, color[0], shape)
    
    return img


def addLabel(highest, lowest, leftmost, rightmost, color, shape):
    # add labels
    label = open(str(color) + "_" + str(shape) + ".txt", "w")
    height = lowest - highest
    width = rightmost - leftmost
    y_center = (lowest + highest)/2
    x_center = (rightmost + leftmost)/2
    label.write("0 " + x_center.astype(str) + " " + y_center.astype(str) + " " + width.astype(str) + " " + height.astype(str))
    label.close
    
    
if __name__ == "__main__":
    # creates images with different shapes and colors
    
    # colors
    g_yellow = ["yellow", [0, 255, 255]]
    g_white = ["white", [255,255,255]]
    g_black = ["black", [0,0,0]]
    g_red = ["red", [0,0, 255]]
    g_blue = ["blue", [255,0,0]]
    g_green = ["green", [0,128,0]]
    g_purple = ["purple", [128,0,128]]
    g_brown = ["brown", [0, 75, 150]]
    g_orange = ["orange", [0,165,255]]
    g_colors = [g_yellow, g_white, g_black, g_red, g_blue, g_green, g_purple, g_brown, g_orange]
    
    # shapes
    g_shapes = ["triangle","rectangle","pentagon", "star"] # TODO: "circle", "semicircle", "quarter circle"
    
    g_size = 100  # size of the shape
    g_background_path = "./src/model/resources/backgrounds/pavement3.jpg"
    g_img = Image.open(g_background_path)
    g_img = np.asarray(g_img)
    
    # initialize directory
    os.chdir("./training_datasets/train/images")
    
    for g_color in g_colors:
        for g_shape in g_shapes:
            coord = (random.randint(0, np.size(g_img, 1)), random.randint(0, np.size(g_img, 0)))
            print("center of the shape: ", coord)
            g_img_with_shape = createImage(g_img, g_shape, coord, g_size, g_color, "A")  
            # g_img_with_shape = Image.fromarray(np.uint8(g_img_with_shape))
            # g_img_with_shape.show()  
    
    cv2.waitKey(0)
