import cv2
cv2.setUseOptimized(True)
import numpy as np
import time

'''
Functions for ball detection in each frame of a video
'''

test = False    # Used to test functions for improvement :)

# create subtracter
fgbg = cv2.createBackgroundSubtractorMOG2(history=15, varThreshold=50, detectShadows=False)
kernel = np.ones((2, 2), np.uint8)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 50
# Filter by Color.
params.filterByColor = True
params.blobColor = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 30
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.75
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.9
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.08

# Create a detector with dos parameters
detector = cv2.SimpleBlobDetector_create(params)


# Finds ball position in orig
def find_ball(frame, height, width):

    '''
    :param frame:
    :param height:
    :param width:
    :return: pos:       2D position of detected ball ([0,0] if none detected)
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # t1 = time.time()
    gray[cv2.medianBlur(fgbg.apply(frame), ksize=5) == 0] = 0
    # t2 = time.time()
    keypoints = detector.detect(gray)
    # t3 = time.time()
    # print('fgbg:'+str(t2-t1))
    # print('detector:'+str(t3-t2))
    col = 0
    row = 0
    if len(keypoints) > 0:

        maxval = 0

        for i in range(len(keypoints)):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])
            val = np.sum(gray[max([y-3,0]):min([y+3,height-1]),max([x-3,0]):min([x+3,width-1])])

            if val > maxval:
                col = x
                row = y
                maxval = val

    pos = np.array([col, row, 1])  # Return the 1 for  calculating 3D pos
    return pos
