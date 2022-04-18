import cv2
import numpy as np
import matplotlib.pyplot as plt
from statistics import median
import math
from scipy import ndimage
import os
import glob


#### Detecting all the points of lines through HoughlinesP
def detect_points(img1):
    resize_img = cv2.resize(cv2.imread(img1), (600, 600))
    gray = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
    img_copy = resize_img.copy()
    edged = cv2.Canny(gray, 30, 200)
    points = cv2.HoughLinesP(edged,
                             1,
                             np.pi / 180,
                             threshold=100,
                             minLineLength=10,
                             maxLineGap=250)
    return points


def detect_slope(lines):
    slopes = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dist = (x2 - x1)
        slope = (y2 - y1) / (x2 - x1)
        slopes.append(slope)
    return slopes


def detect_angles(slopes):
    median_slope = median(slopes)
    angle_in_radians = math.atan(median_slope)
    angle_in_degrees = math.degrees(angle_in_radians)
    return angle_in_degrees


def save_output(img_arr, angle_in_degrees, img_name):
    rotate_image = ndimage.rotate(img_arr, angle_in_degrees)
    cv2.imwrite("output" + img_name + '.jpg', rotate_image)


path = glob.glob("*.jpg")
for img in path:
    img_name = img.split('.')[0]
    img_array = cv2.imread(img)
    points = detect_points(img)
    slopes = detect_slope(points)
    angle = detect_angles(slopes)
    save_output(img_array, angle, img_name)
