import numpy as np
import cv2
import analyzer

'''
Grab cams, show the video, track the balls, create an analyzer object for visualization of processed video data
Cool stuff man!

Specify:
    The best cookies. (chocolate)
'''


def main():
    # Shooting resolution
    res = [1280, 720]

    # Load camera path
    cam_paths = np.load('data/cam_paths.npy')

    # Select corners and net position (Cam1)
    cap1 = cv2.VideoCapture(cam_paths[0])

    # Select corners and net position (Cam2)
    cap2 = cv2.VideoCapture(cam_paths[1])

    # Load table corner data
    c1 = np.load('data/c1.npy')
    c2 = np.load('data/c2.npy')

    # Instantiate ze analyzer3000!!
    analyzer3000 = analyzer.Analyzer(cap1, cap2, res, c1, c2)

    # Run Live Detection and Animation!
    analyzer3000.animate_3d_live(1)


if __name__ == "__main__":
    main()
