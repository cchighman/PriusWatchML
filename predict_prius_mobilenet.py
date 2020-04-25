# Copyright © 2019 by Spectrico
# Licensed under the MIT License
# Based on the tutorial by Adrian Rosebrock: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# Usage: $ python car_make_model_classifier_yolo3.py --image cars.jpg

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import classifier
import shutil
import json
import threading
import queue


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", default='yolo-coco',
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

car_color_classifier = classifier.Classifier()

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
        
def save_result(result, file):
    with open("mobilenet.json") as json_file:
        data = json.load(json_file) 
        temp = data
    
        # python object to be appended 
        y = {
                "confidence": result[0]["confidence"],
                "make": result[0]["make"],
                "model": result[0]["model"],
                "model_year": result[0]["model_year"],
                "file": file
        }
                                                            
        temp.append(y)
    write_json(data)         

def handle_result(image, file,x,y,h,w):
    result = car_color_classifier.predict(image[max(y,0):y + h, max(x,0):x + w])
    print(str(result[0]))
    if "Prius" in str(result[0]['model']) or "Outlander" in str(result[0]['model']):
        print("Image " + file + " - Interesting Result: " + str(result[0]))
        save_result(result, file)
        if os.path.exists("./frames/" + file):
            shutil.move("./frames/" + file, "./interesting/" + file)  
    else:
        if os.path.exists("./frames/" + file):
            shutil.move("./frames/" + file, "../cars/a02/"+ file)

def predictImage(image, file,net):
	# load our input image and grab its spatial dimensions
	image = cv2.imread(image)
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
			if confidence > args["confidence"]:
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
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

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
			   handle_result(image,file,x,y,h,w)
def predict():     
    while True:
        try:
            file = q.get()
            net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
            predictImage("./frames/" + file, file, net)            
        except Exception as e:
            print(e)
        q.task_done()                
                        
def start_predicting():    
    print("Starting Predictions")
    
    procs = 10
    for i in range(0, procs):
        process = threading.Thread(target=predict)
        threads.append(process)

    for t in threads:
        
        t.start()

   
q = queue.Queue()
threads = []

start_predicting()
     
arr = os.listdir("./frames/")
for file in arr:
	if file.endswith("jpg"):
         q.put(file)
         

     

