import numpy as np
import commonDataStructure as cds
from pyproj import Geod


def pixel_to_gps(pixel_x, pixel_y, heading, latitude, longitude, altitude):
    """
    Calculates the GPS location of a pixel

    @param pixel_x: X-axis coordinate, 0 at left
    @param pixel_y: Y-axis coordinatem 0 at top
    @param heading: Azimuth starting at north going clockwise
    @param longitude: Longitude from telemetry data
    @param latitude: Latitude from telemetry data
    @param altitude: Altitude in meters from telemetry data
    @return: (Longitude, latitude) of pixel
    """
    # in pixels
    image_w = 5472
    image_h = 3648
    center_x = image_w // 2
    center_y = image_h // 2

    # in cm
    sensor_w = 1.31328

    # sensor_h = 0.87552
    focal_length = 1.2

    # convert meters to centimeters
    altitude *= 100

    # ground sample distance cm/pixel
    gsd = (altitude * sensor_w) / (focal_length * image_w)

    # distance from center in pixels (left of center negative, below center negative y)
    dist_x = pixel_x - center_x
    dist_y = center_y - pixel_y

    # distance from center in cm
    dist_x *= gsd
    dist_y *= gsd

    # distance from center in m
    dist_x /= 100
    dist_y /= 100

    # rotate point according to heading
    heading_rad = heading * np.pi / 180
    rotation_matrix = np.array(
        [[np.cos(heading_rad), np.sin(heading_rad)],
         [-np.sin(heading_rad), np.cos(heading_rad)]]
    )
    dist_east, dist_north = rotation_matrix @ [[dist_x], [dist_y]]

    # add distances to current gps
    geod = Geod(ellps="WGS84")
    longitude, latitude, _ = geod.fwd(lons=longitude, lats=latitude, az=90, dist=dist_east)
    longitude, latitude, _ = geod.fwd(lons=longitude, lats=latitude, az=0, dist=dist_north)

    return latitude, longitude


def add_dead_reckoning(heading, speed, latitude, longitude, telemetry_time, image_time, weight):
    """
    Estimates current position from speed. Assumes speed has been constant between image and telemetry time

    @param heading: Azimuth starting at north going clockwise
    @param speed: Speed in m/s from telemetry data
    @param latitude: Latitude from telemetry data
    @param longitude: Longitude from telemetry data
    @param telemetry_time: Timestamp of telemetry data in milliseconds
    @param image_time: Timestamp of image in milliseconds
    @param weight: Strength of dead reckoning estimation
    @return: Estimated (longitude, latitude)
    """
    geod = Geod(ellps="WGS84")

    delta_t = telemetry_time - image_time
    displacement = (delta_t / 1000) * speed
    displacement *= weight
    longitude, latitude, _ = geod.fwd(lons=longitude, lats=latitude, az=heading, dist=displacement)


def print_gps_corners(heading, latitude, longitude, altitude):
    """
    Prints GPS positions of the corners of the area covered by camera

    @param heading: Azimuth starting at north going clockwise
    @param latitude: Latitude from telemetry data
    @param longitude: Longitude from telemetry data
    @param altitude: Altitude in meters from telemetry data
    """
    top_left = pixel_to_gps(0, 0, heading, latitude, longitude, altitude)
    top_right = pixel_to_gps(5472, 0, heading, latitude, longitude, altitude)
    bottom_right = pixel_to_gps(5472, 3648, heading, latitude, longitude, altitude)
    bottom_left = pixel_to_gps(0, 3648, heading, latitude, longitude, altitude)
    print(f"{top_left[0]}, {top_left[1]}")
    print(f"{top_right[0]}, {top_right[1]}")
    print(f"{bottom_right[0]}, {bottom_right[1]}")
    print(f"{bottom_left[0]}, {bottom_left[1]}")


# print_gps_corners(15, 49.260605, -123.245995, 25)

# print(pixel_to_gps(5000, 2000, -15, 49.260605, -123.245995, 25))
# def R(theta):
#     """
#     Rotation matrix of theta degrees counterclockwise.
#     """
#
#     theta = np.radians(theta)
#
#     return np.array([[np.cos(theta), -np.sin(theta)],
#                      [np.sin(theta), np.cos(theta) ]])


# def imgCoordToGPS(obj_label: cds.ObjectLabel, drone_info: cds.DroneInfo):
#     """
#     Convert coordinates of an object on an image to coordinates on the GPS.
#     Remember to input the correct numbers or else it may output wrong numbers.
#     Pitch and roll of the drone is assumed to be 0.
#
#     Args:
#         obj_label: info about the object
#         drone_info: info about the drone
#     """
#
#     # extract info from obj_label and drone_info
#     img_h = obj_label.imgh
#     img_w = obj_label.imgw
#     img_x, img_y = [(obj_label.pt1[0] + obj_label.pt2[0]) / 2,
#                     (obj_label.pt1[1] + obj_label.pt2[1]) / 2]
#     theta = drone_info.theta
#     h = drone_info.h
#     FOV_x = drone_info.FOV_x
#     GPS_x = drone_info.x
#     GPS_y = drone_info.y
#
#     # meter (for example, could be inches, feet etc.) to pixel ratio
#     mpxr = img_w / (2 * h) * np.cot(FOV_x / 2)
#
#     # convert image coordinates so that the origin is at the center, the y-axis points up, and the x-axis points right
#     x_1 = img_x - img_w / 2
#     y_1 = img_h / 2 - img_y
#
#     # rotate to align with the GPS coordinate axes
#     P_2 = R(theta) @ np.array([[x_1], [y_1]])
#     x_2 = P_2[0]
#     y_2 = P_2[1]
#
#     # convert pixels to meters (or inches, feet etc.)
#     x_2 = x_2 * mpxr
#     y_2 = y_2 * mpxr
#
#     # translate to align with the origin of the GPS coordinate system
#     x_3, y_3 = [x_2 + GPS_x, y_2 + GPS_y]
#
#     return [x_3, y_3]
