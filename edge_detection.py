<<<<<<< HEAD
import cv2
import numpy as np
import argparse
import sys, time
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
=======
import numpy as np
import math, time, argparse, sys, os
import cv2
>>>>>>> update

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
    return gray

<<<<<<< HEAD
def gaussianBlur(image, size, sigma):
    start = time.time()
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) * normal
    kern_size, gauss = kernel.shape[0], np.zeros_like(image, dtype=float)
    end = time.time()
    print('pre-processing', end-start)
    start = time.time()
    for i in range(image.shape[0] - (kern_size - 1)):
        for j in range(image.shape[1] - (kern_size - 1)):
	    win = image[i:i + kern_size, j:j + kern_size] * kernel
	    gauss[i,j] = np.sum(win)
    end = time.time()
    print('post-processing', end-start)
    return gauss

def cannyFilter1(image):
	gradX, gradY = np.zeros_like(image, dtype=float), np.zeros_like(image, dtype=float)
	amplitude = np.zeros_like(image, dtype=float)
	#applaying filter to image
	for col in range(1, image.shape[0] - 1):
		for row in range(1, image.shape[1] - 1):
			gradX[col, row] = - image[col - 1, row - 1] \
			- image[col, row - 1] \
			- image[col + 1, row -1] \
			+ image[col - 1, row + 1] \
			+ image[col, row + 1] \
			+ image[col + 1, row + 1] \

	    		gradY[col, row] = - image[col - 1, row - 1] \
 	    		- image[col, row - 1] \
	    		- image[col + 1, row - 1] \
	    		+ image[col - 1, row + 1] \
	    		+ image[col, row + 1] \
	    		+ image[col + 1, row + 1]
			amplitude[col, row] = abs(gradX[col, row]) + abs(gradY[col, row])
	    		#amplitude[col, row] = math.sqrt(gradX[col, row]**2 + gradY[col, row]**2)
    	phase = ((np.arctan(gradX/gradY)) / np.pi) * 180
    	phase[phase < 0] += 180
	return amplitude, phase
def cannyFilter(image):
    kernel, kernel_size = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), 3
    gradX, gradY = np.zeros_like(image, dtype=float), np.zeros_like(image, dtype=float)
    #applaying filter to image
    for i in range(image.shape[0] - (kernel_size - 1)):
        for j in range(image.shape[1] - (kernel_size - 1)):
            win = image[i:i + kernel_size, j:j + kernel_size]
            gradX[i, j], gradY[i, j] = np.sum(win * kernel.T), np.sum(win * kernel)

    amplitude = np.sqrt(gradX**2, gradY**2)
    phase = ((np.arctan(gradX/gradY)) / np.pi) * 180
    phase[phase < 0] += 180
    return amplitude, phase

def edgeSupression(amplitude, phase, kernel_size, lowBound, highBound):
    win = np.copy(amplitude)
    for i in range(phase.shape[0] - (kernel_size - 1)):
        for j in range(phase.shape[1] - (kernel_size - 1)):
            if(phase[i, j] <= 22.5 or phase[i, j] >= 157.5):
	        if(amplitude[i,j] <= amplitude[i - 1, j] and amplitude[i,j] <= amplitude[i + 1, j]):
		     amplitude[i, j] = 0
	    if(phase[i, j] > 22.5 or phase[i, j] <= 67.5):
		if(amplitude[i,j] <= amplitude[i - 1, j - 1] and amplitude[i,j] <= amplitude[i + 1, j + 1]):
		     amplitude[i, j] = 0
	    if(phase[i, j] > 67.5 or phase[i, j] <= 112.5):
		if(amplitude[i,j] <= amplitude[i + 1, j + 1] and amplitude[i,j] <= amplitude[i - 1, j - 1]):
		    amplitude[i, j] = 0
	    if(phase[i, j] > 112.5 or phase[i, j] <= 157.5):
		if(amplitude[i,j] <= amplitude[i + 1, j - 1] and amplitude[i,j] <= amplitude[i - 1, j + 1]):
		    amplitude[i, j] = 0
    #treshholding edges
    #strong = np.copy(amplitude)
    #strong[strong < highBound] = 0
    #strong[strong > highBound] = 1
    
    
    M, N = amplitude.shape
    res = np.zeros((M,N), dtype=np.int32)

    highBound = highBound * amplitude.max()
    lowBound = highBound * lowBound 

    strong = np.int32(255)
    strong_i, strong_j = np.where(amplitude >= highBound)
    zeros_i, zeros_j = np.where(amplitude < lowBound)

    res[strong_i, strong_j] = strong 
    return res


def applyFilters(image):
	start = time.time()
        grayMask = rgb2gray(image)
	end = time.time()
	print('grayMask', end - start)
#	start = time.time()
#        blurMask = gaussianBlur(grayMask, 3, 2)
#	end = time.time()
#	print('blurMask', end - start)
	start = time.time()
        cannyAmplitude, cannyPhase = cannyFilter1(grayMask)
	end = time.time()
	print('cannyMask', end - start)
#        edgesStrong = edgeSupression(cannyAmplitude, cannyPhase, 5, 0.05, 0.09)

#        lineImage = np.copy(image) * 0
 #       lines = cv2.HoughLinesP(edgesStrong, 1, np.pi/180, 15, np.array([]), 50, 20)
#
 #       for line in lines:
  #             for x1, y1, x2, y2 in line:
   #                    cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 5)
#
 #       lineEdges = cv2.addWeighted(image, 0.8, lineImage, 1, 0)

        return cannyAmplitude

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', '--mode', help='Mode selection <video / webcam / picture>', required=True)
	args = parser.parse_args()

	if args.mode == 'picture':
	    cap = cv2.imread('testFile/testPicture.png')
	    print('loaded')
	    Mask = applyFilters(cap)
	    cv2.imshow('output', Mask)
            k = cv2.waitKey(0)
            if k == 27:
		cv2.destroyAllWindows()
	elif args.mode == 'video':
	    cap = cv2.VideoCapture('testFile/test_video.mp4')
	    while(cap.isOpened()):
		ret, frame = cap.read()
		#Mask = applyFilters(frame)
                cv2.imshow('output', frame)
                if cv2.waitKey(50) & 0xFF == 27:
                   break
	    cap.release()
	elif args.mode == 'webcam':
	    cap = cv2.VideoCapture(0)
	    while (True):
		ret, frame = cap.read()
                #Mask = applyFilters(frame)
                cv2.imshow('output', frame)
		#time.sleep(0.04)
	   	if cv2.waitKey(1) & 0xFF == 27:
		   break
	    cap.release()
	else:
		print('Invalid argument')
		sys.exit(2)

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main(sys.argv[1:])
=======
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
                        - image[col, row - 1] \
                        - image[col + 1, row - 1] \
                        + image[col - 1, row + 1] \
                        + image[col, row + 1] \
                        + image[col + 1, row + 1]
                        amplitude[col, row] = abs(gradX[col, row]) + abs(gradY[col, row])
                        #amplitude[col, row] = math.sqrt(gradX[col, row]**2 + gradY[col, row]**2)
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
                        

def applyFilters(image):
    cv2.imshow('cv: caption', image/255)
    startTotal = time.time()
    start = time.time()
    gray = rgb2gray(image)
    end = time.time()
    print('time gray:', end - start)
    #apply blur filter
    start = time.time()
    #blur = blurFilter(gray)
    end = time.time()
    print('time blur:', end - start)
    #detect edges
    start = time.time()
    amplitude, phase = edgeFilter(gray)
    end = time.time()
    print('time edge:', end - start)
    #supress edges
    start = time.time()
    supresed = supression(amplitude, phase)
    end = time.time()
    print('time supression:', end - start)
    #tresholding
    start = time.time()
    bounds, weakEdges, strongEdges = tresholding(supresed, 0.2, 0.45)
    end = time.time()
    print('time treshold:', end - start)
    #hysteresis
    start = time.time()
    hyster = hysteresis(bounds, weakEdges, strongEdges)
    end = time.time()
    print('time hysteresis:', end - start)
    
    end = time.time()
    print('===========================')
    print('total time: ', end - startTotal, '\n')
    return hyster

def main(argv):
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--mode', help='Mode selection <video / webcam / picture>', required=True)
        args = parser.parse_args()

        if args.mode == 'picture':
            cap = np.float32(cv2.imread('testFile/kosoctverec.jpg'))
            print('loaded')
            Mask = applyFilters(cap)
            cv2.imshow('output', Mask)
            k = cv2.waitKey(0)
            if k == 27:
                cv2.destroyAllWindows()
        elif args.mode == 'video':
            cap = cv2.VideoCapture('testFile/testVideo.mp4')
            
            while(cap.isOpened()):
                ret, frame = cap.read()
                Mask = applyFilters(frame)
                cv2.imshow('output', Mask)
                if cv2.waitKey(50) & 0xFF == 27:
                   break
            cap.release()
        elif args.mode == 'webcam':
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320);
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180);
            while (True):
                ret, frame = cap.read()
                Mask = applyFilters(frame)
                cv2.imshow('output', Mask)
                if cv2.waitKey(1) & 0xFF == 27:
                   break
            cap.release()
        else:
                print('Invalid argument')
                sys.exit(2)

        cv2.destroyAllWindows()
            
if __name__ == "__main__":
        main(sys.argv[1:])



>>>>>>> update
