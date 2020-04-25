import json
import re
import queue
import time
import threading 
from fastai import *
from fastai.vision import *
from PIL import Image

import requests

import numpy as np
import argparse
import time
import cv2
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


learn = load_learner("./", "resnet50_color.pkl")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-y", "--yolo", default='yolo-coco',
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


def write_json(data, filename="mobilenet.json"): 
    with open(filename,"w") as f: 
        json.dump(data, f, indent=4) 
        
def save_result(prediction, file):
    print("Saving...")
    with open("mobilenet.json") as json_file:
        data = json.load(json_file) 
        temp = data
    
        # python object to be appended 
        y = {
                "category": str(prediction[0]),
                "class": str(prediction[1]),
                "prob": str(prediction[2][prediction[1]]),                
                "image": file
        }
                                                            
        temp.append(y)
    write_json(data)         

def predictCar(stra, file):
    try:
        img = open_image(stra)
        prediction = learn.predict(img)

        print("img: " + file + " label: " + str(prediction[0]) + "  class: " + str(prediction[1]) + "  prob: " + str(format(prediction[2][prediction[1]].item(), '.15f')))
        
        if "prius_blue" in str(prediction[0]):
            print("Image " + file + " - Interesting Result: " + str(prediction[0]))
            save_result(prediction, file)
            os.rename("./frames/" + file, "./interesting/" + file)  
        else:
            os.rename("./frames/" + file, "./frames2/"+ file)
    except:
        os.remove("./frames/" + file)


def detectCars(img,file, net):
    image = cv2.imread(img)
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs = net.forward(output_layers)
    end = time.time()

    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
        0.3)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            if classIDs[i] == 2:
                predictCar(img, file)
    else:
        print("Removing file: " + file)
        os.remove("./frames/" + file)

def predict():     
    while True:
        try:
            file = q.get()
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            print("Detecting..")
            detectCars("./frames/" + file, file, net)            

        except Exception as e:
            if os.path.exists("./frames/"+ file):
                os.remove("./frames/"+file)

        net = None            
        q.task_done()                
                        
def start_predicting():    
    print("Starting Predictions")
    
    procs = 2
    for i in range(0, procs):
        process = threading.Thread(target=predict)
        threads.append(process)

    for t in threads:
        
        t.start()

   
q = queue.Queue()
threads = []

start_predicting()
     
count = 0
arr = os.listdir("./frames/")
for file in arr:
	if file.endswith("jpg"):
         q.put(file) 

     

