import numpy as np
import cv2

'''
Find the port numbers associated with the camera you want to use, data is saved
'''


# CAMERA PATHS
cam_path = 0

# Selected ports (Should only be two)
ports = []

# Shooting resolution
res = (427, 240)


# Run cam path selected
while True:

    if cam_path > 10:
        raise Exception("To many false paths! (Maybe there arent 2 cameras available)")

    # grab frame (Cam1)
    cap = cv2.VideoCapture(cam_path)

    while True:

        # Read frame
        ret1, frame = cap.read()
        frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)

        if not ret1:
            print(f"Something is wrong with path {cam_path}!")
            cam_path += 1
            break

        # Show
        cv2.imshow(f'Camera Path {cam_path}', frame)

        # button press
        k = cv2.waitKey(20) & 0xFF
        if k == 13:

            # This path was good!
            ports.append(cam_path)
            print(ports)

            # If two cameras selected, exit and save data
            if len(ports) == 2:

                # Save data and quit
                np.save('data/cam_paths', np.array(ports))
                print("Data Saved!")

                cv2.destroyAllWindows()
                cap.release()
                quit()

            cam_path += 1
            break  # breaks out of inner loop

        # This path was not good!
        if k == 27:
            cam_path += 1
            break  # breaks out of inner loop

    # Reset for next camera
    cv2.destroyAllWindows()
    cap.release()