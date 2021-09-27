import numpy as np
import math
from scipy import signal    # For signal.gaussian function
import cv2
from myImageFilter import myImageFilter

def angleDecider(angle):
    if (22.5 <= angle and angle < 67.5): return 45
    if (67.5 <= angle and angle < 112.5): return 90
    if (112.5 <= angle and angle < 157.5): return 135
    else: return 0

def NMS(img, row, col, angle):
    if angle == 0: # EAST, WEST
        if ((col-1 >= 0 and img[row, col-1] > img[row,col]) or
            (col+1 < np.shape(img)[1] and img[row, col+1] > img[row,col])):
            img[row,col] = 0
    elif angle == 135: # NORTHEAST, SOUTHWEST
        if ((row-1 >= 0 and col+1 < np.shape(img)[1] and img[row-1, col+1] > img[row,col]) or
            (row+1 < np.shape(img)[0] and col-1 >= 0 and img[row+1, col-1] > img[row,col])):
            img[row, col] = 0
    elif angle == 90: # NORTH, SOUTH
        if ((row-1 >= 0 and img[row-1, col] > img[row,col]) or
            (row+1 < np.shape(img)[0] and img[row+1, col] > img[row,col])):
            img[row,col] = 0
    elif angle == 45: # NORTHWEST, SOUTHEAST
        if ((row-1 >= 0 and col-1 >= 0 and img[row-1, col-1] > img[row,col]) or
            (row+1 < np.shape(img)[0] and col+1 < np.shape(img)[1] and img[row+1, col+1] > img[row,col])):
            img[row,col] = 0
    else:
        raise ValueError("magnitude direction calculating error")
    return img

def myEdgeFilter(img0, sigma):
    # smoothening img0
    hsize = 2 * math.ceil(3 * sigma) + 1
    gaussianFilter = np.reshape(signal.gaussian(hsize, sigma),(hsize,1))
    smoothImg = myImageFilter(img0, gaussianFilter)

    # Sobel Filter
    xSobel = [
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ] 
    ySobel = [
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ]

    imgX = myImageFilter(smoothImg,xSobel)
    imgY = myImageFilter(smoothImg,ySobel)
    imgXY = np.sqrt(imgX**2 + imgY**2)
    # non-maximum suppression
    thetas = (np.arctan2(imgY, imgX) * 180 / np.pi + 180 ) % 180
    for row in range(0, np.shape(thetas)[0]):
        for col in range(0, np.shape(thetas)[1]):
            gratitude_direction = angleDecider(thetas[row,col])
            imgXY = NMS(imgXY, row, col, gratitude_direction)
    return imgXY
