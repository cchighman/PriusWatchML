import numpy as np
import time
from datetime import timedelta

import multiprocessing
from multiprocessing import Pool
import cv2
import os
import copy
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
	help="path to input image")
ap.add_argument("-y", "--yolo", default='yolo-coco',
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

def image_has_shade(boundaries, image):
	for (lower, upper) in boundaries:
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
	
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	return output.any() > 0

def has_required_shades(image):
	shadeList = []
	for shade in requiredShades:
		shadeList.append(image_has_shade(shade, image))
	return all(shadeList)

confidenceVal = 0.4
thresholdVal = 0.3
  
dullBlueGrey = [
#8db1b7 <-> 50757f
	([127,117,80], [183,177,141])]

darkBlueShades = [
#56a3be <-> 84bac8
	([190,163,86], [200,186,133])]

lightBlueShades = [
#d3e8eb <-> 84bac8
	([200,186,133], [235,232,222])]

requiredShades = []
requiredShades.append(dullBlueGrey)
requiredShades.append(darkBlueShades)
requiredShades.append(lightBlueShades)# construct the argument parse and parse the arguments

labelsPath = os.path.sep.join(["./yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
dtype="uint8")# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["./yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["./yolo-coco", "yolov3.cfg"])# load our YOLO object detector trained on COCO dataset (80 classes)

print("[INFO] loading YOLO from disk..." + weightsPath)

def predictImage(image,file,net):
	# load our input image and grab its spatial dimensions

	(H, W) = image.shape[:2]	# determine only the *output* layer names that we need from YOLO
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	outputs = net.forward(output_layers)
	end = time.time()	# show timing information on YOLO
	#print("[INFO] YOLO took {:.6f} seconds".format(end - start))	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []	# loop over each of the layer outputs
	hasShades = False
	for output in outputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > confidenceVal:
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
				idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceVal,thresholdVal)				
    			# ensure at least one detection exists
				if len(idxs) > 0:
					hasShades = False
					# loop over the indexes we are keeping
					for i in idxs.flatten():
						# extract the bounding box coordinates
						(x, y) = (boxes[i][0], boxes[i][1])
						(w, h) = (boxes[i][2], boxes[i][3])						
      					
           				# draw a bounding box rectangle and label on the image
						color = [int(c) for c in COLORS[classIDs[i]]]
						if classIDs[i] == 2:
							text = ""
							if has_required_shades(image[max(y,0):y + h, max(x,0):x + w]):
								#text = "{}: {:.4f}".format(result[0]['make'], float(result[0]['prob']))
								text = "{}".format('Match')
								cv2.putText(image, text, (x + 2, (y - h - 20)), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
								cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
								hasShades = True
		if hasShades is True:
			print("Interesting Image: " + file)
			cv2.imwrite("out_" + file, image)
            
def predict(file):
	try:
		image = cv2.imread(args["path"] +file)		
		if has_required_shades(image) is True:				
			net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)		
			predictImage(image, file, net)
        
	except Exception as e:
		print(e)
			
	print("Removing File: " + file)		
	if os.path.exists(args["path"] + file):
		os.remove(args["path"] + file)  		

def start_predicting():
	print("Processor Count: " + str(multiprocessing.cpu_count()))
	print("Populating images")
	arr = os.listdir(args["path"])
	for file in arr:
		if file.endswith("jpg"):
			images.append(file)

	print("Images populated.  Images: " + str(len(images)))	
	
def start_pool():
    print("Starting Pool")
    p = Pool(8)
    p.map(predict, images)
    
images = []

procs = 4
if __name__ == '__main__':
	start_predicting()
	start_pool()