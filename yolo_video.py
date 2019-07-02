#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 03:31:50 2019

@author: rohitgupta
"""

import numpy as np
import argparse
import cv2
import os
import time
import imutils

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", help = "path to input video", required=True)
# ap.add_argument("-o", "--output", help="path to output video", required=True)
ap.add_argument("-y", "--yolo", help="base path to YOLO directory", required=True)
ap.add_argument("-c", "--confidence", help="minimum probability to filter weak detections", 
                type=float, default=0.5)
ap.add_argument("-t", "--threshold", help="threshold when applying non-maxima suppression", 
                type=float, default=0.3)
args = vars(ap.parse_args())



# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLOURS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype="uint8")


# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)
    

####    Now we’re ready to start processing frames one by one:
    
while True:
    (grabbed, frame) = vs.read()
    
    # if the frame was not grabbed, then we have reached the end
	# of the stream
    if not grabbed:
        break
    
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    
    
    #### let’s perform a forward pass of YOLO, using our current frame  as the input:
    
    # construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB = True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    
    
    # initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
			# of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            # filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # derive top-left coordinates
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    
    # apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            #extract the bouding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLOURS[classIDs[i]]]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

    cv2.imshow("images", frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
    	break
 
	# if the 'q' key is pressed, stop the loop
            

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
