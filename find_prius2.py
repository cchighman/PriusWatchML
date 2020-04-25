import argparse
import copy
import multiprocessing
import os
import queue
import threading
import time
from datetime import timedelta
from multiprocessing import Pool

import cv2
import numpy as np
from PriusImage import PriusImage
from PriusPalette import PriusPalette

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

shadePredictedCount = 0
perfectMatchCount = 0
avgPredictedCount = 0
pcaPredictedCount = 0
totalPredictedCount = 0
shadeSampleCount = 0
avgSampleCount = 0
pcaSampleCount = 0
totalSampleCount = 0

confidenceVal = args["confidence"]
thresholdVal = args["threshold"]

labelsPath = os.path.sep.join(["./yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split(	"\n")  # initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")  # derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["./yolo-coco", "yolov3.weights"])
configPath = os.path.sep.join(["./yolo-coco", "yolov3.cfg"])  # load our YOLO object detector trained on COCO dataset (80 classes)

print("[INFO] loading YOLO from disk..." + weightsPath)


def predictImage(image, file, net):
	# load our input image and grab its spatial dimensions
	global shadePredictedCount
	global avgPredictedCount
	global pcaPredictedCount
	global totalPredictedCount
	global perfectMatchCount

	totalPredictedCount = totalPredictedCount + 1
	(H, W) = image.shape[:2]  # determine only the *output* layer names that we need from YOLO
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	outputs = net.forward(output_layers)
	end = time.time()  # show timing information on YOLO
	# print("[INFO] YOLO took {:.6f} seconds".format(end - start))	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []  # loop over each of the layer outputs
	hasPerfectMatch = False
	hasShades = False
	hasAvgMatch = False
	hasPcaMatch = False
	hasShadeMatch = False
	processed = False

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
				idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceVal, thresholdVal)
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

							start = time.time()
							try:
								priusImage = PriusImage.from_image(image[max(y, 0):y + h, max(x, 0):x + w])

								if priusImage.has_required_shades():
									hasShadeMatch = True
									hasAvgMatch = priusImage.has_avg_match()
									hasPcaMatch = priusImage.has_pca_match()
									hasPerfectMatch = priusImage.has_perfect_match()

									end = time.time()
									text = "{}".format('Match')
									cv2.putText(image, text, (x + 2, (y - h - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
									cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
							except Exception as e:
								print(e)

	if hasShadeMatch is True:
		shadePredictedCount = shadePredictedCount + 1
		cv2.imwrite("shade_" + file, image)

	if hasPcaMatch is True:
		cv2.imwrite("pca_" + file, image)
		pcaPredictedCount = pcaPredictedCount + 1

	if hasAvgMatch is True:
		cv2.imwrite("avg_" + file, image)
		avgPredictedCount = avgPredictedCount + 1

	if hasPerfectMatch is True:
		cv2.imwrite("perfect_" + file, image)
		perfectMatchCount = perfectMatchCount + 1

	print("\n" + str(file) + "\n\t-> Required Palettes: " + str(shadePredictedCount) + "  Average Color: " + str(
		avgPredictedCount) + "  PCA Colors: " + str(pcaPredictedCount) + "  Perfect Match: " + str(
		perfectMatchCount) + "  Total: " + str(totalPredictedCount) + "  Time: " + str(end - start))

def predict():
	while True:
		try:
			file = q.get()
			if file is None:
				break

			priusImage = PriusImage.from_path(args["path"] + file)
			if priusImage.has_required_shades() is True:
				net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
				# print("Checking " + file)
				predictImage(priusImage.image, file, net)

		except Exception as e:
			print(e)


# print("Removing File: " + file)
# if os.path.exists(args["path"] + file):
#	os.remove(args["path"] + file)
# q.task_done()

def start_predicting():
	print("Processor Count: " + str(multiprocessing.cpu_count()))
	procs = 8
	for i in range(0, procs):
		process = threading.Thread(target=predict)
		threads.append(process)
		q.put(None)

	for t in threads:
		t.start()

q = queue.Queue()
threads = []

arr = os.listdir(args["path"])
arr.sort(reverse=True)
print("Populating images")
for file in arr:
	if file.endswith("jpg"):
		q.put(file)

if __name__ == '__main__':
	start_predicting()
