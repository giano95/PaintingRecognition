import cv2
import numpy as np
import imutils

from utils import *

def draw_paintings_bounding_boxes(frame, bounding_boxes):

    for box in bounding_boxes:
        rect, _, _ = grab_painting_pts(box)
        rect = rect.reshape(4,1,2).astype(np.int32)
        cv2.drawContours(frame, [rect], 0, GREEN_COLOR, 2)


def painting_detect(frame):

    # Convert to grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # apply a bilateral filter to denoise the image while preserving the edge
    blurred = cv2.bilateralFilter(src=gray, d=11, sigmaColor=17, sigmaSpace=17)

    # Apply the adaptive threshold to find the edge
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)
    
    # Dilate the edge so finding contours is easier
    dilated = cv2.dilate(src=edges, kernel=np.ones((9, 9), dtype="int"))
    
    # Find contours and sort it by contour area
    contours = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:15]

    # fill the finded contours
    filled = dilated
    for contour in contours:

        epsilon = 0.01 * cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        cv2.drawContours(filled, [contour], 0, WHITE_COLOR, -1)
    
    # re-find the contours and sort it by contour area
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:15]

    paintings_pts = []

    for contour in contours:

        epsilon = 0.1 * cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        polygon = cv2.approxPolyDP(hull, epsilon, True)
        area = cv2.contourArea(hull)

        if area > 25000 and len(polygon) == 4:

            paintings_pts.append(polygon)
    return paintings_pts