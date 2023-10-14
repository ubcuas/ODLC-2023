import numpy as np
import math
import cv2


# ---------------------------- variables----------------------------
# percentage threshold to check if contour area and rectangle area is significantly different
thresh_rect_area = 0.05
# thresh to check if two contours are the same or different.
thresh_centroid_distance = 30
thresh_area_diff = 1000
thresh_match = 0.03
# if the distance between 2 vertices is less than 5, discard one of them.
thresh_distance = 5
# delete the vertex whose angle with the adjacent vertex is wider than a certain degree
thresh_angle = 10
# approximate polygon, epsilon is a percentage of arclength
percentage_of_arclength = 0.02
# delete small shape (noise) and shape unreasonablly large
minimum_percentage_of_img_size = 0.00001
maximum_percentage_of_img_size = 0.8
# ----------------------------end of variables----------------------------


class PolygonContainer:
    """Detect Specific Shape

    Attributes:
        name (str): Name of the shape
        vertex_count (int): vertex number of the shape, 0 means any number
        check (func(contour, approx) -> bool): check function for this shape
        contours (list): shape contour list
        centroids (list): shape centroid list
    """

    def __init__(self, name, vertex_count, check):
        self.name = name
        self.vertex_count = vertex_count
        self.check = check
        self.contours = []
        self.centroids = []


def rectangleCheck(contour, approx):
    """Check if shape is a rectangle

    Args:
        contour (np.ndarray): contour
        approx (np.ndarray): approx of contour

    Returns:
        bool: if contour and approx is the same shape
    """
    # calculate contour area
    contour_area = cv2.contourArea(contour)

    # calculate minimum rectangle area
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    box_area = cv2.contourArea(box)

    # If contour area and rectangle area is significantly different, the shape is not rectangle
    if math.fabs(contour_area - box_area) / contour_area > thresh_rect_area:
        return False
    # If corner of the shape is significantly different from Pi/2, the shape is not rectangle
    for vertex_idx in range(0, len(approx)):
        vec_a = approx[vertex_idx - 1][0] - approx[vertex_idx][0]
        vec_b = approx[(vertex_idx + 1) % len(approx)][0] - \
            approx[vertex_idx][0]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        # check the difference betwween angle and Pi/2 is more than 10 degree
        cos = np.inner(vec_a, vec_b) / (norm_a * norm_b)
        if cos > math.cos((90 - thresh_angle) * math.pi / 180) or cos < math.cos(
            (90 + thresh_angle) * math.pi / 180
        ):
            return False

    # if the shape is rectangle
    return True


def filterRepeatedContours(contours, centroids):
    """Filter the repeated contours and corresponding cnetroids

    Args:
        contours (list): contour list
        centroids (list): centroid list

    Returns:
        contours: filtered contour list
        centroids: filtered centroid list
    """
    # Debug: output all data related to contour
    # for current_contour in range(len(contours)):
    #     print("Contour: %d" % current_contour)
    #     print("Vertex Count: {0}".format(contours[current_contour].shape[0]))
    #     print("Centroid: {0}".format(centroids[current_contour]))
    #     print("Area: {0}".format(cv2.contourArea(contours[current_contour])))
    #     print()

    # use [1.center point distance, 2.shape similarity, 3. area difference] to check if there are repeats in the contour list
    is_valid = np.ones(len(contours), dtype=bool)
    area = [cv2.contourArea(current_contour) for current_contour in contours]
    for current_contour in range(len(contours)):
        if is_valid[current_contour]:
            current_area = area[current_contour]
            for next_contour in range(current_contour + 1, len(contours)):
                vec = centroids[current_contour] - centroids[next_contour]
                distance = np.linalg.norm(vec)
                next_area = area[next_contour]
                area_diff = math.fabs(current_area - next_area)
                match = cv2.matchShapes(
                    contours[current_contour], contours[next_contour], 1, 0.0)
                if distance < thresh_centroid_distance and area_diff < thresh_area_diff and match < thresh_match:
                    print("next_contour removed")
                    is_valid[next_contour] = False
    contours = [contours[current_contour] for current_contour in range(
        len(contours)) if is_valid[current_contour]]
    centroids = [centroids[current_contour]
                 for current_contour in range(len(centroids)) if is_valid[current_contour]]

    # return the filtered contours and centroids
    return contours, centroids


def filterContourVertices(contour):
    """remove repeated vertices in the contour

    Args:
        contour (np.ndarray): contour

    Returns:
        contour: filtered contour
    """

    # delete the close-by vertices
    is_valid = np.ones(contour.shape[0], dtype=bool)
    for current_vertex in range(0, len(contour)):
        if is_valid[current_vertex]:
            for next_vertex in range(current_vertex + 1, len(contour)):
                vec = contour[current_vertex] - contour[next_vertex]
                distance = np.linalg.norm(vec)
                # if the distance between 2 vertices is less than 5, discard one of them.
                if distance < thresh_distance:
                    print("next_vertex removed")
                    is_valid[next_vertex] = False
    contour = contour[is_valid, :]

    # delete the vertex which has wide angle with the adjacent vertex
    is_valid = np.ones(contour.shape[0], dtype=bool)
    for current_vertex in range(0, len(contour)):
        vec_a = contour[current_vertex - 1] - contour[current_vertex]
        vec_b = contour[(current_vertex + 1) %
                        len(contour)] - contour[current_vertex]
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        # delete the vertex whose angle with the adjacent vertex is wider than a certain degree
        cos = np.inner(vec_a, vec_b) / (norm_a * norm_b)
        if cos < math.cos(math.pi * (180 - thresh_angle) / 180):
            is_valid[current_vertex] = False
    contour = contour[is_valid, :]

    # return filtered contour
    return contour


def detectPolygon(img, binarized, approxs, *polygonContainers):
    """Polygon Detection

    Args:
        img (np.ndarray): Original Image
        denoised (np.ndarray): denoised image
        approxs (PolygonContainer): Polygon Container
        polygonContainers (list): Polygon Container List
    """

    # Contour Detection
    contours, _ = cv2.findContours(
        binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.copy(img)
    cv2.drawContours(img_contours, contours, -1, (0, 0, 0), 2)
    cv2.namedWindow("Contours", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Contours", img_contours)

    # Calculate Contour Centroid
    centroids = []
    for current_contour in contours:
        mu = cv2.moments(current_contour, False)
        if np.isclose(mu["m00"], 0):
            mc = contours[0][0]
        else:
            mc = [mu["m10"] / mu["m00"], mu["m01"] / mu["m00"]]
        mc = np.intp(mc)
        centroids.append(mc)

    # Delete repeated Contours
    print("number of centroids: ", len(centroids))
    contours, centroids = filterRepeatedContours(contours, centroids)

    # calculate the image size
    print(img.shape[0])
    print(img.shape[1])
    img_size = img.shape[0] * img.shape[1]

    # approximate polygon for contour
    for current_contour_idx in range(0, len(contours)):

        # get current contour vertices and centroid
        current_contour = contours[current_contour_idx]
        current_centroid = centroids[current_contour_idx]

        # delete small shape (noise) and shape unreasonablly large
        if cv2.contourArea(current_contour) < img_size * minimum_percentage_of_img_size or cv2.contourArea(current_contour) > img_size * maximum_percentage_of_img_size:
            continue

        # approximate polygon
        epsilon = percentage_of_arclength * \
            cv2.arcLength(current_contour, True)
        approx = cv2.approxPolyDP(current_contour, epsilon, True)

        # filter Contour Vertices
        approx = filterContourVertices(approx)

        # Save contour vertices and centroids
        approxs.contours.append(approx)
        approxs.centroids.append(current_centroid)

        # Debug: Disply current polygon and corresponding info
        # showPolygonContours("Approx: %d - vertex_count: %d" % (cid, len(approx)), img, [approx], [current_centroid])
        # print("Approx: %d" % cid)
        # print("Vertex Count: {0}".format(len(approx)))
        # print("Centroid: {0}".format(current_centroid))
        # print("Area: {0}".format(cv2.contourArea(approx)))
        # print("Vertices: {0}".format(approx))
        # print()

        # check if the polygon type is want we want to detect
        vertex_count = len(approx)
        for container in polygonContainers:
            if vertex_count == container.vertex_count and container.check(current_contour, approx):
                container.contours.append(current_contour)
                container.centroids.append(current_centroid)
                print(current_centroid)

    return approxs, polygonContainers


def polygonDetector(img, binarized):
    """Detect Polygon
    
    Args:
        img: original image
        binarized: binarized image
    """
    
    # Detect Polygon
    approxs = PolygonContainer(
        "ApproxPolygons", 0, lambda contour, approx: True)
    triangles = PolygonContainer("Triangle", 3, lambda contour, approx: True)
    rectangles = PolygonContainer("Rectangle", 4, lambda contour, approx: True)
    crosses = PolygonContainer("Cross", 12, lambda contour, approx: True)
    pentagons = PolygonContainer("Pentagon", 5, lambda contour, approx: True)
    stars = PolygonContainer("Star", 10, lambda contour, approx: True)
    # rectangles = PolygonContainer("Rectangle", 4, rectangleCheck)
    detected_approxs, detected_polygoncontainers = detectPolygon(
        img, binarized, approxs, triangles, rectangles, crosses, pentagons, stars)
    
    return detected_approxs, detected_polygoncontainers

