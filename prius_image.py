import os
import cv2
import numpy as np
import time
import datetime
import argparse

from PriusPalette import PriusPalette

from sklearn.cluster import KMeans

class PriusImage:        
    def __init__(self,src):        
        self.src = src
        self.image = cv2.imread(src)
    
    def has_shade(self,boundaries):
    	for (lower, upper) in boundaries:
    		lower = np.array(lower, dtype = "uint8")
    		upper = np.array(upper, dtype = "uint8")
    	
    	mask = cv2.inRange(self.image, lower, upper)
    	output = cv2.bitwise_and(self.image, self.image, mask = mask)
    	return output.any() > 0

    def has_required_shades(self):
    	shadeList = []
    	palette = PriusPalette()
    	for shade in palette.required_shades():
    		shadeList.append(self.has_shade(shade))
    	return all(shadeList)

    def avg_color(self):
        
        height, width, _ = np.shape(self.image)

        # calculate the average color of each row of our image
        avg_color_per_row = np.average(self.image, axis=0)

        # calculate the averages of our rows
        avg_colors = np.average(avg_color_per_row, axis=0)

        # avg_color is a tuple in BGR order of the average colors
        # but as float values
        #print(f'avg_colors: {avg_colors}')

        # so, convert that array to integers
        int_averages = np.array(avg_colors, dtype=np.uint8)
        #print(f'int_averages: {int_averages}')

        # create a new image of the same height/width as the original
        average_image = np.zeros((height, width, 3), np.uint8)
        # and fill its pixels with our average color
        average_image[:] = int_averages

        # finally, show it side-by-side with the original
        #cv2.imshow("Avg Color", np.hstack([img, average_image]))
        #cv2.waitKey(0)
        return int_averages

    def pca_colors(self):
        
        height, width, _ = np.shape(self.image)

        # reshape the image to be a simple list of RGB pixels
        image = self.image.reshape((height * width, 3))

        # we'll pick the 5 most common colors
        num_clusters = 3
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

        return rgb_values
    
    def make_histogram(self,cluster):
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
