import pyqtgraph.opengl as gl
import numpy as np


class RobotArm:
    # ------------------------------
    # Overview
    # ------------------------------

    # This will be the class responsible for the following
    # - Adding render of arm to frame
    # - Calculating joint angles
    # - Looking cool

    # ------------------------------
    # User Variables
    # ------------------------------

    base_joint_height_off_ground = None
    arm_one_length = None
    arm_two_length = None
    paddle_length = None
    distance_from_table = 10  # Inches

    # Init method (in inches)
    def __init__(self, base_height, arm1len, arm2len, paddle_len):
        # Set vars
        self.base_joint_height_off_ground = base_height
        self.arm_one_length = arm1len
        self.arm_two_length = arm2len
        self.paddle_length = paddle_len

    # Render Creator
    def init_arm(self, w):
        initPoints = np.array([(0, 0, 0), (0, 0, 0)])  # init at 0, 0, 0
        line = gl.GLLinePlotItem(pos=initPoints, width=1)
        w.addItem(line)

        return line