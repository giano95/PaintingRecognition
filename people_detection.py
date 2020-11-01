import numpy as np
import imutils
import time
import cv2
from utils import grab_painting_pts
import pathlib

current_path = pathlib.Path().absolute()
weightspath = str(current_path) + "/yolo-coco/yolov3.weights"
configpath = str(current_path) + "/yolo-coco/yolov3.cfg"
net = None
ln = None

def init_yolo():

	global net
	global ln

	net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	if not net or not ln:
		return False
	else:
		return True


def draw_peoples_bounding_boxes(frame, boxes, confidences):

	# apply non-maxima suppression to suppress weak, overlapping bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)
	# ensure at least one detection exists
	if len(idxs) > 0 :
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# draw a bounding box rectangle and label on the frame
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
			text = "person {}".format(i)
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


def people_detect(frame, bounding_boxes_paintings):
	
	global net
	global ln

	H, W = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.3 and classID == 0:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# check if bounding boxes of people is inside a painting
				is_valid = True
				for box in bounding_boxes_paintings:
					rect, dst, (maxWidth, maxHeight) = grab_painting_pts(box)
					if int(rect[0][0]) < int(x) and int(rect[0][1]) < int(y) and maxWidth > width and maxHeight > height:
						is_valid = False
				if is_valid:
					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

	return boxes, confidences
