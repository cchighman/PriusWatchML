import pytest
from datetime import datetime, timedelta
import cv2
from abc import ABC, ABCMeta, abstractmethod
from PriusObjectDetection import PriusPredictor
import copy

path1 = "./time_id.jpg"
newPath1 = "./laterTime_id.jpg"

path2 = "./car1.jpg"
image_hash = '068309071b5b49490096090f2530ffff'
expected_hash = '068309071b5b49490096090f2530ffff'


class Foo(cv2.dnn_Net):
	def __init__(self, param1, param2):
		self._base_params = [param1, param2]
		super(Foo, result).__init__(*self._base_params)

	def __copy__(self):
		cls = self.__class__
		result = cls.__new__(cls)
		result.__dict__.update(self.__dict__)
		super(Foo, result).__init__(*self._base_params)
		return result

	def __deepcopy__(self, memo):
		cls = self.__class__
		result = cls.__new__(cls)
		memo[id(self)] = result
		for k, v in self.__dict__.items():
			setattr(result, k, copy.deepcopy(v, memo))
		super(Foo, result).__init__(*self._base_params)
		return result


@pytest.fixture(scope="module")
def prius_predictor():
	return PriusPredictor()


def test_deep_copy_cnn(prius_predictor):
	net = prius_predictor.darknet_dnn()
	net2 = Foo("a","b")
	print(net)
	netCopy = copy.deepcopy(net)
	assert net == netCopy
