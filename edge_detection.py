import cv2
import numpy as np
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b) / 255
    return gray

def gaussianBlur(image, size, sigma, brightnessAdjust):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) / normal
    kern_size, gauss = kernel.shape[0], np.zeros_like(image, dtype=float)
    #allpaying filter to image
    for i in range(image.shape[0] - (kern_size - 1)):
        for j in range(image.shape[1] - (kern_size - 1)):
	    win = image[i:i + kern_size, j:j + kern_size] * kernel
	    gauss[i,j] = np.sum(win) / brightnessAdjust
    return gauss

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
    weak, strong = np.copy(amplitude), np.copy(amplitude)
    weak[weak < lowBound] = 0
    weak[weak > highBound] = 0

    strong[strong < highBound] = 0
    strong[strong > highBound] = 1
    return weak, strong



image = cv2.imread('testPicture2.png')

grayMask = rgb2gray(image)
blurMask = gaussianBlur(grayMask, 5, 1.5, 200)
cannyAmplitude, cannyPhase = cannyFilter(grayMask)
edgesWeak, edgesStrong = edgeSupression(cannyAmplitude, cannyPhase, 5, 0.09, 0.1)

#lineImage = np.copy(image) * 0
#lines = cv2.HoughLinesP(edgeMask, 1, np.pi/180, 15, np.array([]), 50, 20)

#for line in lines:
#	for x1, y1, x2, y2 in line:
#		cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 5)

#lineEdges = cv2.addWeighted(image, 0.8, lineImage, 1, 0)


#cv2.imshow('grayMask', edgesWeak)
#cv2.imshow('gaussianBLur', edgesStrong)
cv2.imshow('test', edgesStrong)
k = cv2.waitKey(0)
if k == 27:
	cv2.destroyAllWindows()
