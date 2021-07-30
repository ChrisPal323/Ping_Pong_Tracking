import numpy as np
import cv2
import analyzer

'''
Grab cams, show the video, track the balls, create an analyzer object for visualization of processed video data
Cool stuff man!

Specify:
    The best cookies. (chocolate)
'''

# Camera paths lol
cam1_path = 1
cam2_path = 2

# Select corners and net position (Cam1)
cap1 = cv2.VideoCapture(cam1_path)

# Select corners and net position (Cam2)
cap2 = cv2.VideoCapture(cam2_path)

# Load table corner datacd
c1 = np.load('data/c1.npy')
c2 = np.load('data/c2.npy')

# Instantiate ze analyzer3000!!
analyzer3000 = analyzer.analyzer(cap1, cap2, c1, c2)

# Run Live Detection and Animation!
analyzer3000.animate_3d_live()
