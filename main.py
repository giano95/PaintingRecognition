import cv2
import numpy as np
import json
import argparse

from frame_pre_processing import pre_process
from painting_detection import painting_detect, draw_paintings_bounding_boxes
from painting_rectification import painting_rectify, rectify_with_matches
from painting_retrieval import dataset_init, painting_retrieve
from people_detection import people_detect, draw_peoples_bounding_boxes, init_yolo
from people_localization import people_localize
from face_eyes_detection import face_detect, init_face_eye


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="name of the input video file")
    ap.add_argument("-o", "--output", required=True, help="name of the output video file without extension")
    args = vars(ap.parse_args())
    filename_in = args['input']
    filename_out = args['output']

    # Load the paintings db
    print('initializing the paintings dataset...')
    if not dataset_init('resources/paintings_db/'):
        print('error: initialization failed')
        return
    print('initialization complete!')

    # load yolo weights
    print('initializing the yolo weights...')
    if not init_yolo():
        print('error: initialization failed')
        return
    print('initialization complete!')

    # initialize face_eye cascade classifier
    print('initializing the face eye cascade classifier...')
    if not init_face_eye():
        print('error: initialization failed')
        return
    print('initialization complete!')


    #filename = '20180206_113800.mp4'
    #filename = '20180206_113600.mp4'
    #filename = 'IMG_9626.MOV'
    #filename = 'VIRB0395.MP4'
    #filename = 'VIRB0400.MP4'
    #filename = 'VID_20180529_113001.mp4'
    #filename = 'GOPR5826.MP4'
    #filename = '20180206_114604.mp4'
    #filename = '20180206_113059.mp4'
    #filename = 'VIRB0392.MP4'
    #filename = 'IMG_2653.MOV'

    # Open the input video file
    cap = cv2.VideoCapture('input/' + filename_in)
    if not cap.isOpened():
        print("Error while opening the video: " + filename_in)

    # Open the output video file
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output/' + filename_out + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))


    # main loop through all the video frame
    while cap.isOpened():

        ret, frame = cap.read()

        # break if the capture no longer has frames
        if not ret:
            break
        
        #frame = pre_process(frame, undistort=False)

        DRAW_PAINTINGS_BOUNDING_BOXES = True
        DRAW_PEOPLES_BOUNDING_BOXES = True

        bounding_boxes = painting_detect(frame)
        
        paintings = painting_rectify(frame, bounding_boxes)
 
        paintings_data = painting_retrieve(paintings)
    

        paintings_data = rectify_with_matches(paintings_data, frame, draw=False)

        bounding_boxes_people, confidences = people_detect(frame, bounding_boxes)

        people_localize(bounding_boxes_people, paintings_data, draw=True)
 
        face_detect(frame, bounding_boxes, bounding_boxes_people, draw=True)
        
        if DRAW_PAINTINGS_BOUNDING_BOXES:
            draw_paintings_bounding_boxes(frame, bounding_boxes)
        if DRAW_PEOPLES_BOUNDING_BOXES:
            draw_peoples_bounding_boxes(frame, bounding_boxes_people, confidences)

        cv2.imshow("frame", frame)
        cv2.waitKey(7)


        out.write(frame)

        break


    # Release video capture and destroy all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
