from polygonDetector import polygonDetector
from ellipseDetector import ellipseDetector
import colorDetector
import cv2
import numpy as np


print("hello World")

# ---------------------------- variables----------------------------
# Denoise methods: "MedianBlur", "GuassBlur"
denoised_method = "GuassBlur"

# Binarize methods: "AdaptiveThreshold","Canny","Binary","BinaryInv":
binarized_method = "BinaryInv"

# the flag indicating whether a specific color will be chosen for detection
is_specific_color = False
color_of_choice = colorDetector.blue

# image path
# # Test for the self-generated picture
img_path = "./src/object_detector_openCV/images/shapes.png"
# # Test for the real image
# img_path = "./src/object_detector_openCV/images/shapes_real.png"

# ----------------------------end of variables----------------------------

def denoise(gray, method):
    """denoise

    Args:
        gray (np.ndarray): gray image
        method (str): method of denoise

    Returns:
        blurred: denoised gray imaged
    """
    # apply dilation
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=1)

    if method == "MedianBlur":
        blurred = cv2.medianBlur(dilation, 5)
        cv2.namedWindow("MedianBlur", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("MedianBlur", blurred)
        return blurred
    elif method == "GuassBlur":
        blurred = cv2.GaussianBlur(dilation, (5, 5), 0)
        cv2.namedWindow("GuassBlur", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("GuassBlur", blurred)
        return blurred
    else:
        print("No such denoise method!")
        return np.copy(gray)


def binarize(gray, method):
    """Convert gray image to binary image

    Args:
        gray (np.ndarray):gray image
        method (str): binarize method

    Returns:
        thresh: binarized image
    """

    if method == "AdaptiveThreshold":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1
        )
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.namedWindow("AdaptiveThreshold",
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("AdaptiveThreshold", thresh)
        return thresh
    elif method == "Canny":
        thresh = cv2.Canny(gray, 50, 150)
        cv2.namedWindow("Canny", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Canny", thresh)
        return thresh
    elif method == "Binary":
        ret, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        cv2.namedWindow("Binary", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Binary", thresh)
        return thresh
    elif method == "BinaryInv":
        ret, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        cv2.namedWindow("BinaryInv", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("BinaryInv", thresh)
        return thresh
    else:
        print("No such binarize method!")
        return np.copy(gray)
    
    
def printShapes(img, approxs, polygonContainers, ellipseContainer):

    # show all the polygon contours
    img_temp = np.copy(img)
    cv2.drawContours(img_temp, approxs.contours, -1, (0, 0, 0), 3)
    for current_centroid in approxs.centroids:
        cv2.circle(img_temp, tuple(current_centroid), 2, (0, 0, 0), 5)
    cv2.namedWindow("AllPolygon", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("AllPolygon", img_temp)

    # only show the targeted polygons and output the total number of each type of polygons
    img_temp_1 = np.copy(img)
    for container in polygonContainers:
        print("container name: ", container.name)
        cv2.drawContours(img_temp_1, container.contours, -1, (0, 0, 0), 3)
        for current_centroid in container.centroids:
            print(current_centroid)
            cv2.circle(img_temp_1, tuple(current_centroid), 2, (0, 0, 0), 5)
            cv2.putText(img_temp_1, container.name, tuple(current_centroid), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2
                        )
        print("{0} Count: {1}".format(container.name, len(container.centroids)))

    # Show circles/ellipses as red circles
    if(ellipseContainer.keypoints):
        blank = np.zeros((1, 1))
        img_temp_1 = cv2.drawKeypoints(
            img_temp_1, ellipseContainer.keypoints, blank, (0, 0,
                                        255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        for keyPoint in ellipseContainer.keypoints:
            cv2.circle(
                img_temp_1, (int(keyPoint.pt[0]), int(keyPoint.pt[1])), 2, (0, 0, 0), 5)
            cv2.putText(img_temp_1, "Ellipse", (int(keyPoint.pt[0]), int(keyPoint.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2
                        )
        print("Number of Circular Blobs: " + str(len(ellipseContainer.keypoints)))

    if(ellipseContainer.errors):
        cv2.drawContours(img_temp_1, ellipseContainer.contours, -1, (0, 0, 0), 3)
        for current_centroid in ellipseContainer.centroids:
                print(current_centroid)
                cv2.circle(img_temp_1, tuple(current_centroid), 2, (0, 0, 0), 5)
                cv2.putText(img_temp_1, "Ellipse", tuple(current_centroid), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 2
                            )
        print("{0} Count: {1}".format("Ellipse", len(ellipseContainer.centroids)))

    cv2.namedWindow("DetectedShapes", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("DetectedShapes", img_temp_1)
    

def main():
    # read and show the original image
    img = cv2.imread(img_path)
    assert img is not None, "file could not be read, check with os.path.exists()"
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Original", img)

    # Convert to gray image
    color_modified_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.namedWindow("Gray", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("Gray", color_modified_img)

    # wheter choose specific color
    if(is_specific_color):
        color_modified_img = colorDetector.colorDetector(img, color_of_choice)
        color_modified_img = cv2.bitwise_not(color_modified_img)
        cv2.namedWindow("ShowChosenColorOnly", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("ShowChosenColorOnly", color_modified_img)
    
    # Denoise
    denoised = denoise(color_modified_img, denoised_method)

    # Binarize
    binarized = binarize(denoised, binarized_method)

    # Detect polygon
    detected_approxs, detected_polygoncontainers = polygonDetector(img, binarized)

    # Detect Ellipse
    detected_ellipse = ellipseDetector(binarized)            

    printShapes(img, detected_approxs,
                detected_polygoncontainers, detected_ellipse)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


print("Byebye World")
