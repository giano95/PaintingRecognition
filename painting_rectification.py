import cv2
import numpy as np
import imutils
import math
from utils import *


def cut_frames(painting):

    H, W, _ = painting.shape

    gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)

    blurred = cv2.bilateralFilter(src=gray, d=11, sigmaColor=17, sigmaSpace=17)

    # Apply the adaptive threshold
    edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 5)

    # Dilate the canny lines so finding contours is easier
    im_out = cv2.dilate(src=edges, kernel=np.ones((1, 1), dtype="int"))

    contours = cv2.findContours(im_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:25]

    painting_aspect_ratio = W / float(H)

    candidates = []
    for contour in contours:
        
        epsilon = 0.01 * cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        polygon = cv2.approxPolyDP(hull, epsilon, True)
        x, y, w, h = cv2.boundingRect(hull)
        contour_aspect_ratio = w / float(h)
        area = w*h


        if area < 0.9*H*W and area > 0.45*H*W and math.isclose(contour_aspect_ratio, painting_aspect_ratio, rel_tol=0.1):
            candidates.append((x, y, w, h))

    def area(elem):
        x, y, w, h = elem
        return w*h
    
    candidates = sorted(candidates, key = area)

    # if we have a candidate replace the painting, otherwise not
    if len(candidates) > 0:
        corners = candidates[0]
        x, y, w, h = corners
        painting = painting[y:y+h, x:x+w]
        #cv2.rectangle(painting, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
   
    return painting


def painting_rectify(frame, bounding_boxes):

    paintings = []
    for box in bounding_boxes:
        src, dst, dsize = grab_painting_pts(box)

        M = cv2.getPerspectiveTransform(src, dst)
        warped_painting = cv2.warpPerspective(frame, M, dsize)
        paintings.append((warped_painting, box))


    return paintings


def rectify_with_matches(old_paintings_data, frame, draw=False):

    new_paintings_data = []

    n_painting = 0
    for data in old_paintings_data:

        painting, box, info = data

        if not info:
            painting = cut_frames(painting)
            new_paintings_data.append((painting, box, None))
            if draw:
                print('painting n.' + str(n_painting) + ': painting not found!')
        else:

            title, room, filename, score = info

            queryImage = cv2.imread('resources/paintings_db/' + filename, 0)
            trainImage = frame

            if queryImage is None:
                print("Error while opening the db image: " + filename)
                return

            surf = cv2.xfeatures2d.SURF_create(1000)

            kp1, des1 = surf.detectAndCompute(queryImage,None)
            kp2, des2 = surf.detectAndCompute(trainImage,None)

            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)

            matches = bf.knnMatch(des1, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < 0.6*n.distance:
                    good.append(m)
            
            #print(len(good))

            #if we haven enough good matches try to do the perspective transform
            if len(good)> 10:
    
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

                h,w = queryImage.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)

                src, dst, dsize = grab_painting_pts(dst)
                H, W = dsize

                original_aspect_ratio = w / float(h)
                # dumb thing to handle ZeroDivisionError
                if float(H) != 0:
                    warped_aspect_ratio =  W / float(H)
                else:
                    warped_aspect_ratio = 1000000000000.0

                if math.isclose(original_aspect_ratio, warped_aspect_ratio, rel_tol=0.1):
                    M = cv2.getPerspectiveTransform(src, dst)
                    painting = cv2.warpPerspective(frame, M, dsize)

            painting = cut_frames(painting)
            new_paintings_data.append((painting, box, info))
            if draw:
                print('painting n.' + str(n_painting) + ': ', filename, title)

        if draw:
            cv2.imshow('painting ' + str(n_painting), painting)
            cv2.waitKey(97)
 
        n_painting += 1

    return new_paintings_data
