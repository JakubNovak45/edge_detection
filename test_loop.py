import numpy as np
import math, time
import cv2

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
    return gray

def blurFilter(image):
	blur = np.zeros_like(image, dtype=float)
	#applaying filter to image
	for col in range(1, image.shape[0] - 1):
		for row in range(1, image.shape[1] - 1):
			blur[col, row] = (image[col - 1, row - 1] \
			+ image[col - 1 , row] \
			+ image[col - 1 , row + 1] \
			+ image[col, row - 1] \
			+ image[col, row] \
			+ image[col, row + 1] \
			+ image[col + 1, row - 1] \
			+ image[col + 1, row] \
			+ image[col + 1, row + 1]) / 9
	return blur

def edgeFilter(image):
	gradX, gradY = np.zeros_like(image, dtype=float), np.zeros_like(image, dtype=float)
	amplitude = np.zeros_like(image, dtype=float)
	#applaying filter to image
	for col in range(1, image.shape[0] - 1):
		for row in range(1, image.shape[1] - 1):
			gradX[col, row] = - image[col - 1, row - 1] \
			+ image[col + 1, row -1] \
			- image[col - 1, row + 1] \
			+ image[col + 1, row + 1] \
			- image[col - 1, row] \
			+ image[col + 1, row]

	    		gradY[col, row] = - image[col - 1, row - 1] \
 	    		-  image[col, row - 1] \
	    		- image[col + 1, row - 1] \
	    		+ image[col - 1, row + 1] \
	    		+ image[col, row + 1] \
	    		+ image[col + 1, row + 1]
			amplitude[col, row] = abs(gradX[col, row]) + abs(gradY[col, row])
    	phase = ((np.arctan2(gradX, gradY)) / np.pi) * 180
    	phase[phase < 0] += 180
	return amplitude, phase

def supression(amplitude, phase):
	supression = np.zeros(amplitude.shape, dtype=float)
	for col in range(1, amplitude.shape[0] - 1):
		for row in range(1, amplitude.shape[1] - 1):
			a = min(1.0, max(0.0, math.tan(phase[col, row])))
			q = 0
			r = 0
			if(phase[col, row] < 45):
				q = a * amplitude[col + 1, row + 1] + (1 - a) * amplitude[col + 1, row]
				r = a * amplitude[col - 1, row - 1] + (1 - a) * amplitude[col - 1, row]
			elif(phase[col, row] < 90):
				q = a * amplitude[col + 1, row + 1] + (1 - a) * amplitude[col, row + 1]
				r = a * amplitude[col - 1, row - 1] + (1 - a) * amplitude[col, row - 1]	
			elif(phase[col, row] < 135):
				q = a * amplitude[col - 1, row + 1] + (1 - a) * amplitude[col, row + 1]
				r = a * amplitude[col + 1, row - 1] + (1 - a) * amplitude[col, row - 1]
			elif(phase[col, row] < 180):
				q = a * amplitude[col - 1, row + 1] + (1 - a) * amplitude[col - 1, row]
				r = a * amplitude[col + 1, row - 1] + (1 - a) * amplitude[col + 1, row]
			
			if(r < amplitude[col, row] and amplitude[col, row] > q):
				supression[col, row] = amplitude[col, row]

	return supression

def tresholding(image, lowBound, highBound):
	output = np.zeros(image.shape, dtype=float)
	strong_col, strong_row = np.where(image >= highBound)
	weak_col, weak_row = np.where((image <= highBound) & (image >= lowBound))
	output[strong_col, strong_row] = np.float(25)
	output[weak_col, weak_row] = np.float(255)
	return output

cap = np.float32(cv2.imread('testFile/kosoctverec.jpg'))
print('loaded')
start = time.time()
gray = rgb2gray(cap)
#blur = blurFilter(cap)
amplitude, phase = edgeFilter(gray)
supression1 = supression(amplitude, phase)
bounds = tresholding(supression1, 0.4, 0.5)
end = time.time()
print('time: ', end - start)
cv2.imshow('cv: amplitude', amplitude)
cv2.imshow('cv: phase', supression1)
cv2.imshow('cv: supression', bounds)
k = cv2.waitKey(0)
if k == 27:
   cv2.destroyAllWindows()




