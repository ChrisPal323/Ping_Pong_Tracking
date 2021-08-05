import cv2
cv2.setUseOptimized(True)
import numpy as np


'''
Functions for ball detection in each frame of a video
'''

# Test Mode
test = True

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
def find_ball(frame, camNum):
    """
    :param frame:
    :param camNum:
    :return: pos:       2D position of detected ball ([0,0] if none detected)
    """

    # Grab shape
    height, width, _ = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray[cv2.medianBlur(fgbg.apply(frame), ksize=5) == 0] = 0
    keypoints = detector.detect(gray)

    if test:
        im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]),
                                              (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    col = 0
    row = 0
    if len(keypoints) > 0:

        maxval = 0

        for i in range(len(keypoints)):
            x = int(keypoints[i].pt[0])
            y = int(keypoints[i].pt[1])

            val = np.sum(gray[max([y - 3, 0]):min([y + 3, height - 1]), max([x - 3, 0]):min([x + 3, width - 1])])

            if val > maxval:
                col = x
                row = y
                maxval = val

    pos = np.array([col, row, 1])  # Return the 1 for  calculating 3D pos
    if test:
        framecopy = np.copy(frame)
        cv2.circle(framecopy, (col, row), 10, color=(0, 255, 0), thickness=4)
        imageStack = cv2.hconcat([framecopy, im_with_keypoints])
        cv2.imshow(f"Final Detections / Keypoints - {camNum}", imageStack)
    return pos