import numpy as np
import cv2
import math
import json
from operator import itemgetter

from utils import grab_painting_pts


def people_localize(bounding_boxes_people, paintings_data, draw=False):

    # load coordinates from json file "coordinates.json"
    with open('resources/coordinates.json', 'r') as fp:
        coordinates = json.load(fp)

    # load the image of the Museum's map
    museum_map = cv2.imread('resources/map.png')
    if museum_map is None:
        print('Error: i can\'t load the museum map')
        return

    # create an array with the score and the room for each painting obtains from painting retrieval
    candidates = []
    for data in paintings_data:
        painting, box, info = data
        # check if we got information about that bounding box
        if info:
            title, room, num, score = info
            candidates.append([score, int(room)])
    # control that a painting is a real positive to assign a room
    if candidates:
        _, best = min(enumerate(candidates), key=itemgetter(1))
        # look if the score is low enough to consider this painting a true positive
        if best[0] < 10.0:
            index = best[1]
        else:
            # initialize an array of 23 elements (22 rooms,but room 0 doesn't exist)
            # to determine which is the most frequent room
            rooms = np.zeros((23,), dtype=int)
            for score, room in candidates:
                rooms[int(room)] += 1
            # get index of Max value within rooms array
            max_value = np.where(rooms == np.amax(rooms))
            # check if there is only a room with most matches or not
            if len(max_value[0]) == 1:
                index = max_value[0][0]
            else:
                # take the room of the painting closer to people
                for i, box in enumerate(bounding_boxes_people):
                    near_index = 0
                    min_distance = 100000000
                    x, y, width, height = box
                    center_people = (x + (width / 2), y + (height / 2))
                    for data in paintings_data:
                        painting, box, info = data
                        # check if we got information about that bounding box
                        if info:
                            title, room, num, score = info
                            rect, dst, (maxWidth, maxHeight) = grab_painting_pts(box)
                            center_paint = (rect[0][0] + (maxWidth / 2), rect[0][1] + (maxHeight / 2))
                            distance = math.sqrt((center_paint[0] - center_people[0]) ** 2 + (center_paint[1] - center_people[1]) ** 2)
                            if distance < min_distance:
                                min_distance = distance
                                near_index = room
                    # There is a possibility that two differents persons are assigned to two differents rooms
                    draw = False
                    print("[INFO] La persona " + str(i) + " <C3><A8> stata assegnata alla stanza: " + str(near_index))
    else:
        draw = False

    if draw:
        top_left_corner = (coordinates[str(index)][0], coordinates[str(index)][1])
        bottom_right_corner = (coordinates[str(index)][2], coordinates[str(index)][3])
        rectangle = cv2.rectangle(museum_map.copy(), top_left_corner, bottom_right_corner, (0, 0, 255), 8)
        cv2.namedWindow("Map", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("Map", rectangle)
        cv2.resizeWindow("Map", int(rectangle.shape[1] / 2), int(rectangle.shape[0] / 2))
        cv2.waitKey(97)

