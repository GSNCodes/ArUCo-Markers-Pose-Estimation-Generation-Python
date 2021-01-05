'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4
'''

import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["camera"].lower() == "true":
	video = cv2.VideoCapture(0)
	time.sleep(2.0)
	
else:
	if args["video"] is None:
		print("[Error] Video file location is not provided")
		sys.exit(1)

	video = cv2.VideoCapture(args["video"])

if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

while True:
	ret, frame = video.read()
	
	if ret is False:
		break


	h, w, _ = frame.shape

	width=1000
	height = int(width*(h/w))
	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
	corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

	detected_markers = aruco_display(corners, ids, rejected, frame)

	cv2.imshow("Image", detected_markers)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
video.release()