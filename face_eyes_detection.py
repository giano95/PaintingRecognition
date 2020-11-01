import numpy as np
import cv2

face_cascade = None
eye_cascade = None

def init_face_eye():

    global face_cascade
    global eye_cascade

    # load face and eye cascade classifier
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('resources/haarcascade_eye.xml')

    # check the correct file loading
    if not face_cascade or not eye_cascade:
        return False
    else:
        return True


def face_detect(frame, bounding_boxes_paintings, bounding_boxes_people, draw=False):

    global face_cascade
    global eye_cascade

    # check if there are people
    if bounding_boxes_people is None:
        print("[INFO] There are no people")
        return False

    # check if there are paintings
    if bounding_boxes_paintings is None:
        print("[INFO] There are no paintings near the person")
        return False

    # detect Faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for index, box_people in enumerate(bounding_boxes_people):

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # check if faces are inside bounding boxes of people
                if x > box_people[0] and y > box_people[1] and x + w < box_people[0] + box_people[2] and y + h < box_people[1] + box_people[3]:

                    if draw:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # detect Eyes
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    eyes = eye_cascade.detectMultiScale(roi_gray)
                    for (ex, ey, ew, eh) in eyes:

                        if draw:
                            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)

                    print("[INFO] The person ", index, "is facing the camera")

        else:
            print("[INFO] The person ", index, "is facing a painting")

