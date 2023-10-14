
import cv2
import numpy as np
import math

# error must be below this to count as an ellipse
error_threshold = 0.1
        
        
class ellipseContainer:
    """Detect Specific Shape

    Attributes:
        name (str): Name of the shape
        vertex_count (int): vertex number of the shape, 0 means any number
        check (func(contour, approx) -> bool): check function for this shape
        contours (list): shape contour list
        centroids (list): shape centroid list
    """

    def __init__(self):
        self.contours = []
        self.centroids = []
        self.errors = []
        self.keypoints = []
        

def ellipseDetectorByBlob(binary, ellipse):
    """Detect Circle and Ellipse
    Args:
        img (np.ndarray): original image
        denoised (np.ndarray): denoised image
    """

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters, tried 1000/1000000
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 1000000

    # Set Circularity filtering parameters, tried 0.8
    params.filterByCircularity = True
    params.minCircularity = 0.79

    # Set Convexity filtering parameters, tried 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.9

    # Set inertia filtering parameters, tried 0.9
    params.filterByInertia = True
    params.minInertiaRatio = 0.6

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    ellipse.keypoints = detector.detect(binary)

    return ellipse


def ellipseDetectorByHoughCircles(img, denoised):
    # Use HoughCircles to detect circle
    circles = cv2.HoughCircles(
        denoised,
        cv2.HOUGH_GRADIENT,
        1,
        30,
        param1=50,
        param2=60,
        minRadius=0,
        maxRadius=0,
    )

    if circles:
        circles = np.uint16(np.around(circles))
        # Draw outline and center for each circle
        img_circles = np.copy(img)
        for i in circles[0, :]:
            # draw outline
            cv2.circle(img_circles, (i[0], i[1]), i[2], (0, 0, 0), 3)
            # draw center
            cv2.circle(img_circles, (i[0], i[1]), 2, (0, 0, 0), 5)
        cv2.namedWindow("Circle", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Circle", img_circles)

        # Output number of circles
        print("Circle Count: {0}".format(circles.shape[1]))


def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x * y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    if np.linalg.det(S3) == 0:
        return []
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    if np.linalg.det(C) == 0:
        return []
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    print(np.concatenate((ak, T @ ak)).ravel())
    return np.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    
    # makes sures coeffs are real
    print("original_coeffs", coeffs)
    new_coeffs = []
    for coeff in coeffs:
        new_coeffs.append(np.real(coeff))
    print("new_coeffs", new_coeffs)
    
    a = new_coeffs[0]
    b = new_coeffs[1] / 2
    c = new_coeffs[2]
    d = new_coeffs[3] / 2
    f = new_coeffs[4] / 2
    g = new_coeffs[5]

    den = b**2 - a * c
    if den > 0:
        print("coeffs do not represent an ellipse: b^2 - 4ac must" " be negative!")
        return None

    # The location of the ellipse centre.
    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2.0 * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi / 2
    phi = phi % np.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(params, npts=100, tmin=0, tmax=2 * np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def ellipseDetectorByRegression(binarized, detected_ellipse):
    # find contours
    contours, hierarchy = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      
    # splitting contours into x (the x-coords) and y (the y-coords)
    x = []
    y = []
    for contour_index in range(len(contours)):
        x_coords = []
        y_coords = []
        for j in range(len(contours[contour_index])):
            [[x_j, y_j]] = contours[contour_index][j]
            x_coords.append(x_j)
            y_coords.append(y_j)
        x.append(x_coords)
        y.append(y_coords)

    # fits an ellipse onto every contour and computes the sum of the algebraic errors squared
    # it then displays the fitted ellipse and the contours
    for contour_index in range(len(x)):
        for i in range(len(x[contour_index])):
            # scale things by a factor of 1/100
            x[contour_index][i] /= 100
            y[contour_index][i] /= 100

        # fit ellipse
        A = fit_ellipse(np.array(x[contour_index]), np.array(y[contour_index]))
        if len(A) == 0 or cart_to_pol(A) == None:
            continue
        x0, y0, ap, bp, e, phi = cart_to_pol(A)
        print("[", contour_index, "]", "the parameters a, b, c, d, e and f are: ", A)

        # compute error
        error = 0
        for i in range(len(x[contour_index])):
            error += (
                A[0] * x[contour_index][i] ** 2
                + A[1] * x[contour_index][i] * y[contour_index][i]
                + A[2] * y[contour_index][i] ** 2
                + A[3] * x[contour_index][i]
                + A[4] * y[contour_index][i]
                + A[5]
            ) ** 2
        print("[", contour_index, "]", "the total error is: ", error)

        if error < error_threshold:
            detected_ellipse.contours.append(contours[contour_index])
            detected_ellipse.centroids.append((math.floor(x0 * 100), math.floor(y0 * 100)))
            detected_ellipse.errors.append(error)   
        
    return detected_ellipse


def ellipseDetector(binarized):
    """Detect ellipse
    
    Args:
        img: original image
        binarized: binarized image
    """
    # Detect ellipse
    detected_ellipseContainers = ellipseContainer()
    
    # # use cv2.SimpleBlobDetector to detect ellipse
    # binarized_inv = cv2.bitwise_not(binarized)
    # ellipseDetectorByBlob(binarized_inv, detected_ellipseContainers)
    
    # using regression to detect ellipse works better when differentiating ellipse from rectangle, but is more sensitive to image quality
    ellipseDetectorByRegression(binarized, detected_ellipseContainers)

    
    return detected_ellipseContainers
