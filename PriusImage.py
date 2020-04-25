import argparse
import datetime
import os
import time

import cv2
import numpy as np
from PriusPalette import PriusPalette
from sklearn.cluster import KMeans


class PriusImage(object):
	def __init__(self, image):
		self.image = image
		self.avgColor = []
		self.pcaColors = []

	@classmethod
	def from_image(cls, image):
		return cls(image)

	@classmethod
	def from_path(cls, path):
		image = cv2.imread(path)
		return cls(image)

	def has_perfect_match(self):
		try:
			palette = PriusPalette()
			isMatch = False
			matches = []
			matches.append(self.has_required_shades())

			avg = self.avg_color()
			matches.append(palette.has_shade(avg))

			requiredColors = []
			for color in self.pca_colors():
				requiredColors.append(palette.has_shade(color))

			matches.append(any(requiredColors))
			return all(matches)
		except Exception as e:
			print(e)
		return False

	def color_profile(self):
		start = time.time()
		try:
			shadePredictedCount = 0
			avgPredictedCount = 0
			pcaPredictedCount = 0
			totalPredictedCount = 0

			# print("\nColor Profile for Image")

			palette = PriusPalette()

			totalPredictedCount = totalPredictedCount + 1
			if self.has_required_shades():
				shadePredictedCount = shadePredictedCount + 1

			avg = self.avg_color()
			avgInPalette = palette.has_shade(avg)
			# print("Average Color: " + str(avg) + "  Palette Match? " + str(avgInPalette))
			if avgInPalette:
				avgPredictedCount = avgPredictedCount + 1

			requiredColors = []
			for color in self.pca_colors():
				requiredColors.append(palette.has_shade(color))
			# print("PCA Analysis\n")
			if any(requiredColors):
				pcaPredictedCount = pcaPredictedCount + 1
		except Exception as e:
			print(e)
		end = time.time()
		print("\nRequired Palettes: " + str(shadePredictedCount) + "  Average Color: " + str(
			avgPredictedCount) + "  PCA Colors: " + str(pcaPredictedCount) + "  Total: " + str(
			totalPredictedCount) + "  Time: " + str(end - start) + "\n")

	def has_shade(self, shade):
		for (lower, upper) in shade:
			lower = np.array(lower, dtype="uint8")
			upper = np.array(upper, dtype="uint8")

		mask = cv2.inRange(self.image, lower, upper)
		output = cv2.bitwise_and(self.image, self.image, mask=mask)
		return output.any() > 0

	def has_required_shades(self):
		shadeList = []
		palette = PriusPalette()
		for shade in palette.required_shades():
			shadeList.append(self.has_shade(shade))
		return all(shadeList)

	def avg_color(self):
		if len(self.avgColor) > 0:
			return self.avgColor

		height, width, _ = np.shape(self.image)

		# calculate the average color of each row of our image
		avg_color_per_row = np.average(self.image, axis=0)

		# calculate the averages of our rows
		avg_colors = np.average(avg_color_per_row, axis=0)

		# avg_color is a tuple in BGR order of the average colors
		# but as float values
		# print(f'avg_colors: {avg_colors}')

		# so, convert that array to integers
		int_averages = np.array(avg_colors, dtype=np.uint8)
		# print(f'int_averages: {int_averages}')

		# create a new image of the same height/width as the original
		average_image = np.zeros((height, width, 3), np.uint8)
		# and fill its pixels with our average color
		average_image[:] = int_averages

		# finally, show it side-by-side with the original
		# cv2.imshow("Avg Color", np.hstack([img, average_image]))
		# cv2.waitKey(0)
		self.avgColor = int_averages
		return int_averages

	def has_avg_match(self):
		palette = PriusPalette()
		avgColor = self.avg_color()

		if palette.has_shade(avgColor):
			return True

		return False

	def has_pca_match(self):
		palette = PriusPalette()
		pcaColors = self.pca_colors()

		for color in pcaColors:
			if palette.has_shade(color):
				return True

		return False

	def pca_colors(self):
		if len(self.pcaColors) > 0:
			return self.pcaColors
		height, width, _ = np.shape(self.image)

		# reshape the image to be a simple list of RGB pixels
		image = self.image.reshape((height * width, 3))

		# we'll pick the 5 most common colors
		num_clusters = 1
		clusters = KMeans(n_clusters=num_clusters)
		clusters.fit(image)

		# count the dominant colors and put them in "buckets"
		histogram = self.make_histogram(clusters)
		# then sort them, most-common first
		combined = zip(histogram, clusters.cluster_centers_)
		combined = sorted(combined, key=lambda x: x[0], reverse=True)

		# finally, we'll output a graphic showing the colors in order
		rgb_values = []
		for index, rows in enumerate(combined):
			rgb = int(rows[1][2]), int(rows[1][1]), int(rows[1][0])
			rgb_values.append(rgb)

		self.pcaColors = rgb_values
		return rgb_values

	def make_histogram(self, cluster):
		"""
		Count the number of pixels in each cluster
		:param: KMeans cluster
		:return: numpy histogram
		"""
		numLabels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
		hist, _ = np.histogram(cluster.labels_, bins=numLabels)
		hist = hist.astype('float32')
		hist /= hist.sum()
		return hist
