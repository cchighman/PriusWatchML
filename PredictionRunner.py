import argparse
import multiprocessing
import os
import shutil
import time
from multiprocessing import Pool

from PriusImage import PriusImage
import cv2
from PriusObjectDetection import PriusPredictor

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=False,
                help="path to input image")
args = vars(ap.parse_args())

images = []
prius = PriusPredictor()


class PriusPredictionRunner(object):

	def __init__(self):
		pass

	def predict(self, image_meta):

		start = time.time()
		for eachObject, eachObjectPath in prius.detect_vehicle(image_meta):
			# print(eachObject["name"], " : ", str(eachObject["percentage_probability"]), " : ",
			#    str(eachObject["box_points"]))

			prediction_meta = dict(image_name=eachObject['name'], image_points=eachObject["box_points"],
			                       image_path=eachObjectPath)

			predictions, probabilities = prius.predict_vehicle(prediction_meta)
			found_prius = False
			prius_prob = ''
			for eachPrediction, eachProbability in zip(predictions, probabilities):
				if "prius" in eachPrediction and int(eachProbability) > 50:
					img = PriusImage.from_path(eachObjectPath)
					hasPCA = img.has_pca_match()
					print("#### PRIUS IDENTIFIED: " + image_meta['image_name'] + " with probability " + str(
						eachProbability) + " PCA: " + str(hasPCA))
					found_prius = True
					prius_prob = eachProbability
				print(image_meta['image_name'] + " -> " + eachPrediction, " : ", eachProbability)
			try:
				if found_prius:
					shutil.copy(os.path.join(image_meta['image_path'], image_meta['image_name']),
					            './frames/detection/match_' + str(prius_prob) + '_' + str(hasPCA) + "_" + image_meta['image_name'])

				shutil.move(os.path.join(image_meta['image_path'], image_meta['image_name']),
				            './frames/processed/' + image_meta['image_name'])

			except Exception as e:
				pass

		end = time.time()
		print("Prediction Time: " + str(end - start))

	def start_pool(self, count):
		print("Starting Pool")
		p = Pool(count)
		p.map(self.predict, images)

def start_predicting():
	print("Processor Count: " + str(multiprocessing.cpu_count()))

	print("Populating images")
	arr = os.listdir(args["path"])
	for file in arr:
		if file.endswith("jpg"):
			images.append(dict(image_path=args["path"], image_name=file))

	print("Images populated.  Images: " + str(len(images)))
	runner.start_pool(8)


runner = PriusPredictionRunner()
if __name__ == '__main__':
	start_predicting()
