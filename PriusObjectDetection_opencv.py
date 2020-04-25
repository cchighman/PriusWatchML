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


class PriusPredictor(object):
	def __init__(self):
		self.avgColor = []
		self.pcaColors = []

		self.confidenceVal = 0.5
		self.threshold = 0.3
		self.yolo_folder = './yolo-coco/'

		self.hasPerfectMatch = False
		self.hasAvgMatch = False
		self.hasPcaMatch = False
		self.hasShadeMatch = False

		self.shadePredictedCount = 0
		self.avgPredictedCount = 0
		self.pcaPredictedCount = 0
		self.totalPredictedCount = 0
		self.perfectMatchCount = 0

		self.shadePredictedCount = 0
		self.perfectMatchCount = 0
		self.avgPredictedCount = 0
		self.pcaPredictedCount = 0
		self.totalPredictedCount = 0
		self.shadeSampleCount = 0
		self.avgSampleCount = 0
		self.pcaSampleCount = 0
		self.totalSampleCount = 0

		self.confidenceVal = 0.5
		self.thresholdVal = 0.4

		# initialize a list of colors to represent each possible class label
		self.labelsPath = os.path.sep.join([self.yolo_folder, "coco.names"])
		self.LABELS = open(self.labelsPath).read().strip().split("\n")
		np.random.seed(42)

		# derive the paths to the YOLO weights and model configuration
		self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")
		self.weightsPath = os.path.sep.join([self.yolo_folder, "yolov3.weights"])

		# load our YOLO object detector trained on COCO dataset (80 classes)
		self.configPath = os.path.sep.join([self.yolo_folder, "yolov3.cfg"])

	def write_prediction_results(self):
		if self.hasShadeMatch is True:
			self.shadePredictedCount = shadePredictedCount + 1
			cv2.imwrite("shade_" + file, image)

		if self.hasPcaMatch is True:
			cv2.imwrite("pca_" + file, image)
			self.pcaPredictedCount = pcaPredictedCount + 1

	if self.hasAvgMatch is True:
		cv2.imwrite("avg_" + file, image)
		self.avgPredictedCount = avgPredictedCount + 1

	if self.hasPerfectMatch is True:
		cv2.imwrite("perfect_" + file, image)
		self.perfectMatchCount = perfectMatchCount + 1

	self.totalPredictedCount = totalPredictedCount + 1


def predict_prius(image, coords):
	prius_image = PriusImage.from_image(
		image[max(coords[1], 0):coords[1] + coords[3], max(coords[0], 0):coords[0] + coords[2]])

	if prius_image.self.has_required_shades():
		self.hasShadeMatch = True
		self.hasAvgMatch = priusImage.self.has_avg_match()
		self.hasPcaMatch = priusImage.self.has_pca_match()
		self.hasPerfectMatch = priusImage.self.has_perfect_match()

		text = "{}".format('Match')
		cv2.putText(image, text, (x + 2, (y - h - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

		print(
			"\n" + str(file) + "\n\t-> Required Palettes: " + str(shadePredictedCount) + "  Average Color: " + str(
				avgPredictedCount) + "  PCA Colors: " + str(pcaPredictedCount) + "  Perfect Match: " + str(
				perfectMatchCount) + "  Total: " + str(totalPredictedCount))


def darknet_dnn(self):
	net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
	return net


def detect_vehicle(self, image):
	# image = meta_data['vehicle_image']
	# image_path = meta_data['image_path']
	image = cv2.imread(image)
	# determine only the *output* layer names that we need from YOLO
	(H, W) = image.shape[:2]
	layer_names = net.getLayerNames()

	# construct a blob from the input image and then perform a forward
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)

	start = time.time()
	outputs = net.forward(output_layers)
	# show timing information on YOLO
	end = time.time()

	boxes = []
	confidences = []
	classIDs = []  # loop over each of the layer outputs

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
			if confidence > self.confidenceVal:
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
				idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidenceVal, self.thresholdVal)
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
							try:
								coords = ([x], [y], [w], [h])
								predict_data = dict(color=color, image=image, coords=coords, image_path=image_path)
								predict_prius(predict_data)
							except Exception as e:
								print(e)
