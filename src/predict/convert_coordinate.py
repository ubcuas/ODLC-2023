import numpy as np
import commonDataStructure as cds


def R(theta):
    """
    Rotation matrix of theta degrees counterclockwise.
    """
    
    theta = np.radians(theta)
    
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta) ]])
    
    
def imgCoordToGPS(obj_label: cds.ObjectLabel, drone_info: cds.DroneInfo):
    """
    Convert coordinates of an object on an image to coordinates on the GPS.
    Remember to input the correct numbers or else it may output wrong numbers.
    Pitch and roll of the drone is assumed to be 0.
    
    Args:
        obj_label: info about the object
        drone_info: info about the drone
    """
    
    # extract info from obj_label and drone_info
    img_h = obj_label.imgh
    img_w = obj_label.imgw
    img_x, img_y = [(obj_label.pt1[0] + obj_label.pt2[0]) / 2,
                    (obj_label.pt1[1] + obj_label.pt2[1]) / 2]
    theta = drone_info.theta
    h = drone_info.h
    FOV_x = drone_info.FOV_x
    GPS_x = drone_info.x
    GPS_y = drone_info.y
    
    # meter (for example, could be inches, feet etc.) to pixel ratio
    mpxr = img_w / (2 * h) * np.cot(FOV_x / 2)
    
    # convert image coordinates so that the origin is at the center, the y-axis points up, and the x-axis points right
    x_1 = img_x - img_w / 2
    y_1 = img_h / 2 - img_y
    
    # rotate to align with the GPS coordinate axes
    P_2 = R(theta) @ np.array([[x_1], [y_1]])
    x_2 = P_2[0]
    y_2 = P_2[1]
    
    # convert pixels to meters (or inches, feet etc.)
    x_2 = x_2 * mpxr
    y_2 = y_2 * mpxr
    
    # translate to align with the origin of the GPS coordinate system
    x_3, y_3 = [x_2 + GPS_x, y_2 + GPS_y]
    
    return [x_3, y_3]
