
import numpy as np
import cv2
from PIL import Image

# in BGR colorspace
yellow = [0, 255, 255]
white = [255,255,255]
black = [0,0,0]
red = [0,0, 255]
blue = [255,0,0]
green = [0,128,0]
purple = [128,0,128]
brown = [0, 75, 150]
orange = [0,165,255]

def getLimits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 30, 100, 100], dtype=np.uint8)
        upperLimit = np.array([190, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 30, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
        
    # if hue >= 165:  # Upper limit for divided red hue
    #     lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    # elif hue <= 15:  # Lower limit for divided red hue
    #     lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    # else:
    #     lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
    #     upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

def colorDetector(img, selected_color):
    assert img is not None, "file could not be read, check with os.path.exists()"
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Original", img)
    cv2.namedWindow("hsvImage", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("hsvImage", hsvImage)
    
    lowerLimit, upperLimit = getLimits(color=selected_color)
    
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)
 
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # cv2.namedWindow("DetectedColor", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.imshow("DetectedColor", img)
    
    return mask

def main():
    img_path = "./python/opencv/shapes_real.jpg"
    img = cv2.imread(img_path)
    colorDetector(img, blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lowerLimit, upperLimit = getLimits(color=purple)
    
#     mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

#     mask_ = Image.fromarray(mask)
 
#     bbox = mask_.getbbox()

#     if bbox is not None:
#         x1, y1, x2, y2 = bbox

#         frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()

# cv2.destroyAllWindows()

