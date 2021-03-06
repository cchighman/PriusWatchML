import argparse
import multiprocessing
import os
import json
import queue
import shutil
import threading
import time
from multiprocessing import Pool

import cv2
from PriusImage import PriusImage
from PriusObjectDetection import PriusPredictor
from imageai.Detection import ObjectDetection
from prius_color import has_prius_color

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False,
                help="path to images",
                default="/content/drive/Colab Notebooks/")
ap.add_argument("-m", "--models", required=False,
                default="./",
                help="path to models")
ap.add_argument("-t", "--threading", required=False,
                default="single",
                help="thread, pool, or single")
ap.add_argument("-o", "--output", required=False,
                help="output path")
ap.add_argument("-p", "--method", required=False,
                default='predict',
                help="only predict")
ap.add_argument("-a", "--accuracy", required=False,
                default=70,
                help="Prius probability accuracy")
ap.add_argument("-r", "--results", required=False,
                default=70,
                help="Prius Results")
ap.add_argument("-z", "--test", required=False,
                help="True for test",default=False)
args = vars(ap.parse_args())

q = queue.Queue()

images = []
threads = []
prius = PriusPredictor(args['images'], args['models'], args['output'])


def write_json(data, filename=args['results'] + "prius_results.json"):
	with open(filename, "w") as f:
		json.dump(data, f, indent=4)


def save_result(result):
	with open(args['results'] + "prius_results.json") as json_file:
		data = json.load(json_file)
		temp = data

		temp.append(result)
	write_json(data)


def get_files(path):
	items = []
	count = 0
	for root, d_names, f_names in os.walk(path):
		for f in f_names:
			if count < 5000:
				items.append(dict(image_path=root, image_name=f))
				count = count + 1
			else:
				return items
	return items


class PriusPredictionRunner(object):

	def __init__(self):
		pass

	def detect_vehicle(self, image_meta):
		detected = prius.detect_vehicle(image_meta)

		if "person" in eachObject["name"] or "car" in eachObject["name"]:
			return detected
		return False

	def predict_vehicle(self, image_meta):
		predictions, probabilities = prius.predict_vehicle_method(image_meta)
		prius_prob = 0
		found = False
		for eachPrediction, eachProbability in zip(predictions, probabilities):
			# print ("Prediction: " + str(eachPrediction) + " - Probabilitiy: " + str(eachProbability))
			if "prius" in eachPrediction and int(eachProbability) > int(args['accuracy']):
				prius_prob = eachProbability
				print("---> Prius Identified: " + image_meta['image_name'] + " with probability " + str(
					eachProbability) + " at path:  " + image_meta['image_path'])

				result = {
					"time": str(time.localtime()),
					"image_name": image_meta["image_name"],
					"image_path": image_meta["image_path"],
					"probability": str(eachProbability),
					"object": str(eachPrediction)
				}

				save_result(result)

				found = True

		return dict(result=found, prob=prius_prob)

	def predict_array(self, decoded):
		start = time.time()
		found_prius = False
		prius_prob = 0
		try:
			prius.detect_vehicle_from_array(decoded)
				#prediction_meta = dict(image_name=eachObject['name'], image_points=eachObject["box_points"],
				                   #    image_path=eachObjectPath)

				#if has_prius_color(eachObjectPath, eachObjectPath):
				#	result_meta = self.predict_vehicle(prediction_meta)
			#		found_prius = result_meta['result']
		#			prius_prob = result_meta['prob']
		'''
		except Exception as e:
			print("Exception while predicting: " + str(e))

		try:
			if found_prius:
				shutil.copy(os.path.join(image_meta['image_path'], image_meta['image_name']),
				            args['output'] + 'detection/match_' + str(prius_prob) + "_" + image_meta['image_name'])

			if args['test'] is False:
				shutil.move(os.path.join(image_meta['image_path'], image_meta['image_name']),
				            args['output'] + 'processed/' + image_meta['image_name'])
'''
		except Exception as e:
			print("Exception while predicting: " + str(e))

		end = time.time()
		print("Prediction Time: " + str(end - start))


	def predict(self, image_meta):
		start = time.time()
		found_prius = False
		prius_prob = 0
		try:
			for eachObject, eachObjectPath in prius.detect_vehicle(image_meta):
				prediction_meta = dict(image_name=eachObject['name'], image_points=eachObject["box_points"],
				                       image_path=eachObjectPath)

				if has_prius_color(eachObjectPath, eachObjectPath):
					result_meta = self.predict_vehicle(prediction_meta)
					found_prius = result_meta['result']
					prius_prob = result_meta['prob']
		except Exception as e:
			print("Exception while predicting: " + str(e))

		try:
			if found_prius:
				shutil.copy(os.path.join(image_meta['image_path'], image_meta['image_name']),
				            args['output'] + 'detection/match_' + str(prius_prob) + "_" + image_meta['image_name'])

			if args['test'] is False:
				shutil.move(os.path.join(image_meta['image_path'], image_meta['image_name']),
				            args['output'] + 'processed/' + image_meta['image_name'])

		except Exception as e:
			print("Exception while predicting: " + str(e))

		end = time.time()
		print("Prediction Time: " + str(end - start))

	def predict_threading(self):
		while True:
			# try:
			image = q.get()
			if image is None:
				break
			self.predict(image)
			#	except Exception as e:
			#	print(e)
			q.task_done()

	def start_pool(self, count):
		print("Starting Pool")
		p = Pool(count)
		p.map(self.predict, images)

	def start_threads(self, count):
		print("Starting Threads")

		for i in range(0, count):
			process = threading.Thread(target=self.predict_threading)
			threads.append(process)

		for t in threads:
			t.start()
			q.put(None)


runner = PriusPredictionRunner()


def start_predicting_pool():
	print("Processor Count: " + str(multiprocessing.cpu_count()))

	print("Populating images")
	for file in get_files(args['images']):
		if "processed" not in file["image_path"] and "detection" not in file["image_path"] and file[
			"image_name"].endswith(".jpg"):
			images.append(file)

	print("Images populated.  Images: " + str(len(images)))
	runner.start_pool(multiprocessing.cpu_count())


def start_predicting_threads():
	print("Multi-Threaded - Processor Count: " + str(multiprocessing.cpu_count()))

	print("Populating images")
	for file in get_files(args['images']):
		if "processed" not in file["image_path"] and "detection" not in file["image_path"] and file[
			"image_name"].endswith(".jpg"):
			q.put(file)

	print("Images populated.")
	runner.start_threads(multiprocessing.cpu_count())


def start_predicting_single():
	print("Single Thread - Processor Count: " + str(multiprocessing.cpu_count()))

	print("Populating images")
	for root, dirs, files in os.walk(args['images']):
		for name in files:
			if "processed" not in name and "detection" not in name and name.endswith(".jpg"):
				if args["method"] == 'detect':
					runner.predict(dict(image_path=root, image_name=name))
				elif args["method"] == 'predict':
					runner.predict_vehicle(dict(image_path=root, image_name=name))


if __name__ == '__main__':
	if args['threading'] == 'pool':
		start_predicting_pool()
	elif args['threading'] == 'thread':
		start_predicting_threads()
	elif args['threading'] == 'single':
		start_predicting_single()
