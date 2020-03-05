## Object detection pipeline not using Machine Learning for initial candidate selection.
## Software reads in all .jpg files in the current working directory.
## Using a gaussian adaptive threshold to identify regions with intensity differences from the local environment.
## Edge and contour detection is used to detect shaped sources in the thresholded areas.
## False positive rejection attempts to eliminate sources caused by extended hot surfaces.

## This version of the code does not record data to file but instead displays the raw image with the bounding boxes.
## Green bounding boxes pass the false positive removal pipeline, red ones fail and would not be saved.

import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import csv
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import matplotlib.pyplot as plt

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

## Parse arguments for the detection pipeline.

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--range", type=int, required=True, help="range of the filter")
ap.add_argument("-o", "--offset", type=int, required=True, help="offset of the filter")
ap.add_argument("-b", "--blur", type=int, required=True, help="size of gaussian blur")
ap.add_argument("-l", "--low", type=int, required=True, help="low size of edge detection")
ap.add_argument("-m", "--max", type=int, required=True, help="max size of edge detection")
args = vars(ap.parse_args())

## Open a list of file names in the current work directory cwd.

imgs = []
cwd = os.getcwd()

## Add .jpg files to the list of file names.

for file in os.listdir(cwd):
	if file.endswith(".jpg"):
		imgs.append(file)

## For each of the images, open the image and execute the detection pipeline.

for file in imgs:
	img = cv2.imread(file)
	ori = img
	orisave = img

	## Convert the image to grayscale and apply a median blur of kernel 5.

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.medianBlur(img,5)

	sizey, sizex = img.shape

	## Read the arguments into variables.

	rang = args["range"]
	os = args["offset"]
	gau = args["blur"]
	low = args["low"]
	high = args["max"]

	## Apply a gaussian threshold with range and offset supplied in the arguments.

	#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, \
        #                   cv2.THRESH_BINARY,rang,os)
	th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                           cv2.THRESH_BINARY,rang,os)

	## Bitwise not the image and apply a gaussian blur of kernel gau.

	image = th
	image = cv2.bitwise_not(image)
	gray = image
	gray = cv2.GaussianBlur(gray, (gau, gau), 0)

	## Perform Canny edge detection with thresholds 50 to 100. (Works well in practice, a bit magic numbery atm).
	## Dilate and Erode the edged lines to clean up noisy edges.

	edged = cv2.Canny(gray, low, high)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	## Find contours in the edges indicating complete objects.

	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	try:
		(cnts, _) = contours.sort_contours(cnts)
	except:
		continue

	pixelsPerMetric = None

	## Open up a count for the number of accepted bounding boxes.

	z = 0

	areas = []

	## For each detected contour run the validation process.

	if len(cnts) > 10:
		continue

	for c in cnts:

		areas.append(cv2.contourArea(c))

		## Reject contours with areas outside of the range 40-400. Again, this is a bit magic numbery here.

		if cv2.contourArea(c) < 40:
			continue

		if cv2.contourArea(c) > 400:
			continue

		## Apply a bounding box around the contour.

		orig = image.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		box = perspective.order_points(box)

		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)

		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		x, y, w, h = cv2.boundingRect(c)

		## Select a Region of Interest from the image inside the bounding box.
		
		roi = image[y:y+h, x:x+w]
		#cv2.rectangle(ori, (x,y), (x+w, y+h), (0, 255, 0), 2)

		## Record dimensions of the bounding box incase of future filtering.

		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		#if abs((dA / dB) - 1) >= 0.99:
		#	continue

		#cv2.drawContours(ori, [box.astype("int")], -1, (0, 255, 0), 2)

		#for (x, y) in box:
			#cv2.circle(ori, (int(x), int(y)), 5, (0, 0, 255), -1)

		#cv2.circle(ori, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		#cv2.circle(ori, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		#cv2.circle(ori, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		#cv2.circle(ori, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

		#cv2.line(ori, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
		#cv2.line(ori, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

		## Legacy code for sizing the objects, not needed.

		if pixelsPerMetric is None:
			pixelsPerMetric = dB / 100

		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric

		## Record a grayscale image of the RoI area.

		roi = orisave[(y):(y+h), (x):(x+w)]
		roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

		## Determine the brighest pixel in the base image.

		maxpix = np.amax(ori)

		## Now for the false positive removal, only do if the RoI contains data.

		if np.amax(roi) != 0:

			## Again, apply Canny edge detection in the RoI.

			edges = cv2.Canny(roi, 100, 200)			
			edges = cv2.dilate(edges, None, iterations=1)
			edges = cv2.erode(edges, None, iterations=1)

			if edges is not None:

				result = edges.copy()

				## If edges are present, record the pixel values for the edges of the RoI.
				## Do not include the corner pixels.

				resshape = result.shape
				ed1 = np.amax(result[1:(resshape[0]-1),0])
				ed2 = np.amax(result[1:(resshape[0]-1),(resshape[1]-1)])
				ed3 = np.amax(result[0, 1:(resshape[1]-1)])
				ed4 = np.amax(result[(resshape[0]-1), 1:(resshape[1]-1)])

				## By default, keep the RoI.

				keep = True

				if np.amax(edges) == 0:

					## Reject if no edges are detected.

					keep = False
					print("Rejecting, edge pixels = 0")
				else:

					## Check for the contour crossing the edges of the RoI.
					## If it does not, keep it, if it does it on all 4 sides, also keep it.
					## This is due to it likely being an object with a slightly too small box.
					## Otherwise, reject it unless the RoI contains a pixel with 0.75x Intensity
					## of the brightest pixel in the base image.

					if 0 in [ed1,ed2,ed3,ed4]:
						if np.sum(np.array([ed1,ed2,ed3,ed4]) != 0) <= 1:
							keep = True
							print("Keeping, border pix = 0 or 1")
						else:
							if np.amax(roi)/maxpix >= 0.75:
								keep = True
								print("Keeping, brightest pixel >= 0.75max")
							else:
								keep = False
								print("Rejecting, too dim pixels")
					else:
						keep = True
						print("Keeping, all borders are 0 pix value")

			else:
				keep = False
				print("Rejecting, no edges detected")

		else:
			keep = False
			print("Rejecting, all ROI pixels are 0")

		## This completes the false positive rejection.

		## Now draw the bounding boxes on the image.

		cv2.imshow('Image', ori)

		if keep is True:
			cv2.rectangle(ori, (x,y), (x+w, y+h), (0, 255, 0), 2)
		else:
			cv2.rectangle(ori, (x,y), (x+w, y+h), (0, 0, 255), 2)

	print("------------------------")
	cv2.waitKey(0)
