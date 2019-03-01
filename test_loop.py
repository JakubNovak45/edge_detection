import numpy as np
import math, time
import cv2

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
    return gray

def blurFilter(image):
	blur = np.zeros_like(image, dtype=float)
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
			if(phase[col, row] < 45 and phase[col, row] > 0):
				q = a * amplitude[col + 1, row + 1] + (1 - a) * amplitude[col + 1, row]
				r = a * amplitude[col - 1, row - 1] + (1 - a) * amplitude[col - 1, row]
			elif(phase[col, row] < 90):
				q = a * amplitude[col + 1, row + 1] + (1 - a) * amplitude[col, row + 1]
				r = a * amplitude[col - 1, row - 1] + (1 - a) * amplitude[col, row - 1]	
			elif(phase[col, row] < 135):
				q = a * amplitude[col - 1, row + 1] + (1 - a) * amplitude[col, row + 1]
				r = a * amplitude[col + 1, row - 1] + (1 - a) * amplitude[col, row - 1]
			elif(phase[col, row] <= 180):
				q = a * amplitude[col - 1, row + 1] + (1 - a) * amplitude[col - 1, row]
				r = a * amplitude[col + 1, row - 1] + (1 - a) * amplitude[col + 1, row]
			else:
				print('you fucked up', phase[col, row])
			if(r < amplitude[col, row] and amplitude[col, row] > q):
				supression[col, row] = amplitude[col, row]
	return supression

def tresholding(image, lowBound, highBound):
	output = np.zeros(image.shape, dtype=float)
	weakEdges = np.float(25)
	strongEdges = np.float(255)
	strong_col, strong_row = np.where(image >= highBound)
	weak_col, weak_row = np.where((image <= highBound) & (image >= lowBound))
	output[strong_col, strong_row] = weakEdges
	output[weak_col, weak_row] = strongEdges
	return output, weakEdges, strongEdges

def hysteresis(image, strongEdges, weakEdges):
	for col in range(1, image.shape[0] - 1):
		for row in range(1, image.shape[1] - 1):
			if(image[col, row] == weakEdges):
				if ((image[col+1, row-1] == strongEdges) or (image[col+1, row] == strongEdges) 
				or (image[col+1, row+1] == strongEdges) or (image[col, row-1] == strongEdges)
				or (image[col, row+1] == strongEdges) or (image[col-1, row-1] == strongEdges) 
				or (image[col-1, row] == strongEdges) or (image[col-1, row+1] == strongEdges)):
                        		image[col, row] = strongEdges
                    		else:
					image[col, row] = 0
	return image		
			

cap = np.float32(cv2.imread('testFile/testPicture1.png'))
print('loaded')
startTotal = time.time()
start = time.time()
gray = rgb2gray(cap)
end = time.time()
print('time gray:', end - start)

start = time.time()
blur = blurFilter(gray)
end = time.time()
cv2.imshow('cv: blur', blur)
print('time blur:', end - start)

start = time.time()
amplitude, phase = edgeFilter(gray)
end = time.time()
print('time edge:', end - start)

start = time.time()
supression1 = supression(amplitude, phase)
end = time.time()
print('time supression:', end - start)

start = time.time()
bounds, weakEdges, strongEdges = tresholding(supression1, 0.2, 0.45)
end = time.time()
print('time treshold:', end - start)

start = time.time()
hysteresis = hysteresis(bounds, weakEdges, strongEdges)
end = time.time()
print('time hysteresis:', end - start)

end = time.time()
print('total time: ', end - startTotal)
cv2.imshow('cv: amplitude', hysteresis)
#cv2.imshow('cv: phase', cap)
#cv2.imshow('cv: hysteresis', hysteresis)
k = cv2.waitKey(0)
if k == 27:
   cv2.destroyAllWindows()




