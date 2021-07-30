import numpy as np
import cv2
import ball_tracking as bt
import time
import os

'''
Calibrate the ping pong table, find the corners and use background subtraction
'''


class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            self.points.append((x, y))


# CAMERA PATHS
cam1_path = 0
cam2_path = 1

# grab frame (Cam1)
cap1 = cv2.VideoCapture(cam1_path)
ret1, frame = cap1.read()
h1, w1, c1 = frame.shape

# Use da class
coordinateStore = CoordinateStore()

# Create a black image, a window and bind the function to window
cv2.namedWindow('Calibrate Table')
cv2.setMouseCallback('Calibrate Table', coordinateStore.select_point)

# Run cam 1
while True:
    cv2.imshow('Calibrate Table', frame)

    # Next Camera if ESC is pressed
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

# Close frames and express data
cv2.destroyAllWindows()
print("Selected Coordinates: ")
c1 = [[p[0], h1 - p[1], 1] for p in coordinateStore.points]
c1 = np.array(c1)
print(c1)
cap1.release()

# grab frame (Cam2)
cap2 = cv2.VideoCapture(cam2_path)
ret2, frame = cap2.read()
h2, w2, c2 = frame.shape

# Use da class once more
coordinateStore = CoordinateStore()

# Create a black image, a window and bind the function to window
cv2.namedWindow('Calibrate Table')
cv2.setMouseCallback('Calibrate Table', coordinateStore.select_point)

# Run cam 2
while True:
    cv2.imshow('Calibrate Table', frame)

    # End if ESC is pressed
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break

# Close frames and express data
cv2.destroyAllWindows()
print("Selected Coordinates: ")
c2 = [[p[0], h2 - p[1], 1] for p in coordinateStore.points]
c2 = np.array(c2)
print(c2)
cap2.release()

# Save info
np.save('data/c1', c1)
np.save('data/c2', c2)
