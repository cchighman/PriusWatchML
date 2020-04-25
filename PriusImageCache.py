import io
import os
import re
import time
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime, timedelta
from time import mktime
from typing import Any, Union

import cv2
import dhash
from PIL import Image


class Deduplication(ABC):
	__metaclass__ = ABCMeta

	# Constructor
	def __init__(self):
		self.cache = dict()

	# Abstract Methods
	@abstractmethod
	def is_image_duplicate(self, src, cam_id=''):
		pass

	@abstractmethod
	def get_image_hash(self, src):
		pass

	# Concrete Methods
	def get_hash_key(self, src):
		return re.search(r'_(.*)\.jpg', src).group()

	@abstractmethod
	def put_hash(self, src, cam_id=''):
		pass

	def get_hash(self, src):
		meta = self.cache[self.get_hash_key(src)]
		return meta['image_hash']

	def compare_images(self, image1, image2):
		hash1 = self.get_image_hash(image1)
		hash2 = self.get_image_hash(image2)

		diff = dhash.get_num_bits_different(int(hash1, 16), int(hash2, 16))

		if diff > 3:
			return False

		return True


class PathDeduplication(Deduplication):

	def put_hash(self, src, cam_id=''):
		image_hash = self.get_image_hash(src)
		meta = dict(timestamp=time.localtime(), image_hash=image_hash)

		self.cache[self.get_hash_key(src)] = meta

	def is_image_duplicate(self, src, cam_id=''):
		meta = self.cache[self.get_hash_key(src)]

		if meta is None:
			return False

		new_hash = self.get_image_hash(src)
		diff = dhash.get_num_bits_different(int(meta['image_hash'], 16), int(new_hash, 16))
		if diff > 3:
			return False
		return True

	def get_image_hash(self, src):
		image = Image.open(src)
		row, col = dhash.dhash_row_col(image)
		return dhash.format_hex(row, col)


class ImageDeduplication(Deduplication):

	def put_hash(self, src, cam_id=''):
		imageStream = io.BytesIO(src)
		imageFile = Image.open(imageStream)

		row, col = dhash.dhash_row_col(imageFile)
		image_hash = dhash.format_hex(row, col)
		meta = dict(timestamp=time.localtime(), image_hash=image_hash)
		print("Putting hash " + str(image_hash) + " for cam " + str(cam_id) + " in cache.")
		self.cache[cam_id] = meta

	def is_image_duplicate(self, src, cam_id=''):
		print("Checking Duplicate: " + str(cam_id))

		if cam_id not in self.cache:
			return False

		meta = self.cache[cam_id]
		new_hash = self.get_image_hash(src)

		diff = dhash.get_num_bits_different(int(meta['image_hash'], 16), int(new_hash, 16))

		if diff > 3:
			return False
		return True

	def get_image_hash(self, src):
		imageStream = io.BytesIO(src)
		imageFile = Image.open(imageStream)
		row, col = dhash.dhash_row_col(imageFile)
		return dhash.format_hex(row, col)
