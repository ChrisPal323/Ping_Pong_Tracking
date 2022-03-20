import sys
import numpy as np
from collections import deque
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
import robot_arm


'''
Rendering Helpers, to make cool visuals!
'''


# Used to init a ball to be moved
# params:
#   w - 3D window
# return:
#   ball object
def init_ball(w):
    center = np.array([0, 0, 0])  # init at 0, 0, 0
    radius = 0.85  # Set radius for ball

    md = gl.MeshData.sphere(rows=10, cols=20, radius=radius)

    b = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=(255, 255, 255, 0.3),
        shader="balloon",
        glOptions="additive",
    )

    b.translate(*center)
    w.addItem(b)

    return b


# Used to init a point (prob for testing
# params:
#   w - 3D window
def init_point(w, x, y, z):
    center = np.array([x, y, z])  # init at 0, 0, 0
    radius = 0.45  # Set radius for ball

    md = gl.MeshData.sphere(rows=10, cols=20, radius=radius)

    point = gl.GLMeshItem(
        meshdata=md,
        smooth=True,
        color=(255, 0, 0, 1)
    )

    point.translate(*center)
    w.addItem(point)


# Used to init a camera
# params:
#   w - 3D window
#   num - camera num
#   posX... - 3D pos of camera
#   dirX... - Vector of direction (to be normalized)
# return:
#   added camera object
def init_camera(w, num, posX, posY, posZ, dirX, dirY, dirZ):

    cameraPos = np.array([posX, posY, posZ])

    cameraTarget = np.array([dirX, dirY, dirZ])  # target dir point
    normDirection = cameraTarget / np.sqrt(np.sum(cameraTarget**2))  # Cool maths!

    # Draw Line
    initPoints = np.array([cameraPos, normDirection])
    arrow = gl.GLLinePlotItem(pos=initPoints, width=1)
    w.addItem(arrow)

    # Draw box as camera (idk bro)
    camera = gl.GLBoxItem()
    camera.setSize(2, 2, 2)
    camera.translate(-1, -1, -1)

    #camera.rotate(45, 1, 0, 0)  # Rotate x degrees around specificed axis (x, y, z)

    camera.translate(*cameraPos)
    w.addItem(camera)

    return camera


# Used to init a path (line) that will be relocated to trace the pos of the ball
# params:
#   w - 3D window
# return:
#   ball object
def init_path(w):

    initPoints = np.array([(0, 0, 0), (0, 0, 0)])  # init at 0, 0, 0
    line = gl.GLLinePlotItem(pos=initPoints, width=1)
    w.addItem(line)

    return line

