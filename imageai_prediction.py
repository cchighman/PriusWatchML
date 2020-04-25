
from imageai.Prediction.Custom import CustomImagePrediction
import os

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath( "model_ex-093_acc-0.818203.h5")
prediction.setJsonPath( "model_class.json")
prediction.loadModel(num_objects=191)

images = os.listdir("./predicted/sample2/")
for image in images:
	if "jpg" in image:
		predictions, probabilities = prediction.predictImage("./predicted/sample2/" + image, result_count=3)
		for eachPrediction, eachProbability in zip(predictions, probabilities):
			print(image + " - " + eachPrediction, " : ", eachProbability)
