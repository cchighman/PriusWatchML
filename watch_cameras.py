import argparse
import json
import os
import queue
import re
import threading
import time
from datetime import timedelta

import PriusImageCache
import numpy as np
import requests
from PriusImageCache import ImageDeduplication
from timeloop import Timeloop

tl = Timeloop()
cams = []

cam_threads = []

dedup = ImageDeduplication()
q = queue.Queue()

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--timer", required=False,
                help="path to input image")
ap.add_argument("-c", "--cams", default='min_cams2.json',
                help="base path to YOLO directory")

args = vars(ap.parse_args())


def watch_camera():
	while True:
		try:
			cam = q.get()
			if cam is None:
				break

			now = time.localtime()
			frame_folder = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + "_" + str(
				now.tm_hour) + "-" + str(now.tm_min) + "/"

			frame_file = str(now.tm_year) + str(now.tm_mon) + str(now.tm_mday) + "_" + str(
				now.tm_hour) + "-" + str(now.tm_min) + "_" + str(cam['id']) + ".jpg"

			frame_dir = pathFolder + frame_folder
			if os.path.exists(frame_dir) is False:
				os.mkdir(frame_dir)
			path = frame_dir + frame_file

			img_data = requests.get(cam['url']).content

			if dedup.is_image_duplicate(img_data, cam['id']):
				print(str(frame_file) + " is a duplicate image.  Removing.")
				if os.path.exists(path):
					os.remove(path)
			else:
				# Update hash
				dedup.put_hash(img_data, cam['id'])

				# Save File
				with open(path, 'wb') as handler:
					handler.write(img_data)
		except Exception as e:
			print(e)

		cam = None
		result = None


@tl.job(interval=timedelta(seconds=int(args["timer"])))
def watch_redmond_cameras_timer():
	print("Cameras job current time : {}".format(time.ctime()))
	for cam in cams:
		q.put(cam)


def start_watching():
	procs = 30
	for i in range(0, procs):
		process = threading.Thread(target=watch_camera)
		cam_threads.append(process)

	for t in cam_threads:
		t.start()


if __name__ == '__main__':
	print("Loading Seattle Cams")
	with open(args["cams"], "r") as read_file:
		cams = json.load(read_file)

	for cam in cams:
		q.put(cam)

	pathFolder = "./frames/"

	tl.start()
	start_watching()
