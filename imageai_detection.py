import os
import time
from PIL import Image as plt
from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join("yolo.h5"))  # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
detector.loadModel()

images = os.listdir("./frames/")
for image in images:
	if "jpg" in image:
		start = time.time()
		try:
			detections, objects_path = detector.detectObjectsFromImage(input_image=image_path + image,
			                                                           extract_detected_objects=True,
			                                                           output_image_path="/content/cars/test/prius/box_" + image,
			                                                           minimum_percentage_probability=30)
			for eachObject, eachObjectPath in zip(detections, objects_path):
				print(eachObject["name"], " : ", str(eachObject["percentage_probability"]), " : ",
				      str(eachObject["box_points"]))
				for detected in os.listdir(eachObjectPath):
					print("Copying " + eachObjecPath + "/" + detected + " to ../" + detected)
					copy.copy(eachObjectPath + "/" + detected, "../" + detected)

		except:
			pass
		end = time.time()
		print("Time: " + str(end-start))
		for eachObject in detections:
			print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])

		plt.imread("./frames/detected_" + image)
		plt.show()
