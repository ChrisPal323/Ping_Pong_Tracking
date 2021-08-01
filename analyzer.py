import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque
from scipy.linalg import qr, svd
import cv2
import random
import matplotlib.animation as animation
import ball_tracking as bt

xtable = 2.74
ytable = 1.525


class Analyzer:
    """
    Analyzer class for applying triangulation, finding 3D tracks, and visualization

    Args:
        height1 (int): Height dimension of camera 1
        height2 (int): Height dimension of camera 2
        width1 (int): Width dimension of camera 1
        width2 (int): Width dimension of camera 2
        corners1 (np.array): Positions of corners and net in
                                camera 1
        corners2 (np.array): Positions of corners and net in
                                camera 2
        ball_pos_1 (np.array): Detected positions of ball in
                                camera 1
        ball_pos_2 (np.array): Detected positions of ball in
                                camera 2

    Attributes:
        h1      Height camera 1
        h2      Height camera 2
        w1      Width camera 1
        w2      Width camera 2
        bp1     Ball position camera 1
        bp2     Ball position camera 2
        pc1     Corner positions camera 1
        pc2     Corner positions camera 2
        c3d     3D corner positions
        P1      Camera matrix 1
        P2      Camera matrix 2
                Factorizations of camera matrices:
        K1
        K2
        A1
        A2
                Normalized camera matrices:
        P1norm
        P2norm
                Detected points etc:
        points
    """

    # Initiate and calculate cameras, etc.
    def __init__(self, cam1, cam2, resolution, corners1, corners2):

        # 3D points of ball
        pts_len = 16
        self.p3d = deque(maxlen=pts_len)

        # Camera objects
        self.cam1 = cam1
        self.cam2 = cam2

        # Set Resolution for cameras
        cam1.set(3, resolution[0])
        cam1.set(4, resolution[1])
        cam2.set(3, resolution[0])
        cam2.set(4, resolution[1])

        self.height = resolution[0]
        self.width = resolution[1]

        # Points in cornersX should correspond: p1-p3, p2-p4, p3-p1, p4-p2
        self.pc1 = np.copy(corners1)
        self.pc2 = np.zeros([6, 3])
        self.pc2[0, :] = corners2[2, :]
        self.pc2[1, :] = corners2[3, :]
        self.pc2[2, :] = corners2[0, :]
        self.pc2[3, :] = corners2[1, :]
        self.pc2[4, :] = corners2[5, :]
        self.pc2[5, :] = corners2[4, :]
        self.pc1 = np.transpose(self.pc1)
        self.pc2 = np.transpose(self.pc2)

        # Calculate camera matrices P1 and P2 from 6 known points
        p1 = [0, ytable, 0, 1]
        p2 = [xtable, ytable, 0, 1]
        p3 = [xtable, 0, 0, 1]
        p4 = [0, 0, 0, 1]
        p5 = [xtable / 2, -0.1525, 0.15, 1]
        p6 = [xtable / 2, ytable + 0.1525, 0.15, 1]
        self.c3d = np.transpose(np.array([p1, p2, p3, p4, p5, p6]))
        # Calculate P1 and P2
        self.P1 = calc_P(self.c3d, self.pc1)
        self.P2 = calc_P(self.c3d, self.pc2)
        [r1, q1] = rq(self.P1)
        [r2, q2] = rq(self.P2)
        self.K1 = r1
        self.K2 = r2
        self.A1 = q1
        self.A2 = q2
        self.P1norm = np.matmul(np.linalg.inv(self.K1), self.P1)
        self.P2norm = np.matmul(np.linalg.inv(self.K2), self.P2)

    # Finds a 3D point from 2 2D points
    def calc_3d_point(self, x1, x2):  # Points should be [a,b,1]
        x1norm = np.matmul(np.linalg.inv(self.K1), x1)
        x2norm = np.matmul(np.linalg.inv(self.K2), x2)
        M = np.zeros([6, 6])
        M[0:3, 0:4] = self.P1norm
        M[3:6, 0:4] = self.P2norm
        M[0:3, 4] = -x1norm
        M[3:6, 5] = -x2norm
        [_, _, V] = np.linalg.svd(M)
        v = V[5, :]
        X = pflat(np.reshape(v[0:4], [4, 1]))
        return np.reshape(X[0:3], [3, ])

    # Plots path taken by ball in 3D in one point
    def plot_3d_point(self, pointnbr):
        fig = plt.figure()
        ax = Axes3D(fig)
        for stroke in self.points[pointnbr - 1]:
            strokecopy = []
            for i in range(stroke.shape[0]):
                if stroke[i, 0] != 0 and stroke[i, 1] != 0:
                    strokecopy.append([stroke[i, 0], stroke[i, 1], stroke[i, 2]])
            sc = np.array(strokecopy)
            ax.scatter([sc[i, 0] for i in range(sc.shape[0])], [sc[i, 1] for i in range(sc.shape[0])],
                       [sc[i, 2] for i in range(sc.shape[0])])
        x = [0, 0, 2.74, 2.74, 0, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37]
        y = [0, 1.525, 1.525, 0, 0, 0, -0.1525, -0.1525, 1.525 + 0.1525, 1.525 + 0.1525, -0.1525]
        z = [0, 0, 0, 0, 0, 0, 0, 0.1525, 0.1525, 0, 0]
        pos1 = -np.matmul(np.linalg.inv(self.A1[0:3, 0:3]), self.A1[:, 3])
        pos2 = -np.matmul(np.linalg.inv(self.A2[0:3, 0:3]), self.A2[:, 3])
        dir1 = self.A1[2, :]
        dir2 = self.A2[2, :]
        ax.scatter(pos1[0], pos1[1], pos1[2], c='k')
        ax.scatter(pos2[0], pos2[1], pos2[2], c='k')
        ax.quiver(pos1[0], pos1[1], pos1[2], dir1[0], dir1[1], dir1[2], length=1, normalize=True)
        ax.quiver(pos2[0], pos2[1], pos2[2], dir2[0], dir2[1], dir2[2], length=1, normalize=True)
        ax.plot(x, y, z, 'r')
        # maxpos = max([np.max(pos1),np.max(pos2)])
        ax.set_xlim(-1, 4)
        ax.set_ylim(-2, 3)
        ax.set_zlim(-1, 3)
        plt.show()


    # Live generation
    def animate_3d_live(self, interval):

        # Config the plot
        fig = plt.figure()
        ax = Axes3D(fig)

        # Set Scales
        ax.set_xlim(-1, 4)
        ax.set_ylim(-2, 3)
        ax.set_zlim(-1, 3)

        def plot_table(ax):
            # Ping Pong Table Coords
            x = [0, 0, 2.74, 2.74, 0, 1.37, 1.37, 1.37, 1.37, 1.37, 1.37]
            y = [0, 1.525, 1.525, 0, 0, 0, -0.1525, -0.1525, 1.525 + 0.1525, 1.525 + 0.1525, -0.1525]
            z = [0, 0, 0, 0, 0, 0, 0, 0.1525, 0.1525, 0, 0]

            # Plot Table
            ax.plot(x, y, z, 'b', linewidth=2)

        def plot_cameras(ax):
            # Set up stuff for drawing camera
            pos1 = -np.matmul(np.linalg.inv(self.A1[0:3, 0:3]), self.A1[:, 3])
            pos2 = -np.matmul(np.linalg.inv(self.A2[0:3, 0:3]), self.A2[:, 3])
            dir1 = self.A1[2, :]
            dir2 = self.A2[2, :]

            # Draw the cameras
            ax.scatter(pos1[0], pos1[1], pos1[2], c='k')
            print(pos1[0], pos1[1], pos1[2])
            ax.scatter(pos2[0], pos2[1], pos2[2], c='k')
            ax.quiver(pos1[0], pos1[1], pos1[2], dir1[0], dir1[1], dir1[2], length=1, normalize=True)
            ax.quiver(pos2[0], pos2[1], pos2[2], dir2[0], dir2[1], dir2[2], length=1, normalize=True)

        def grab_frames():
            ret1, frame1 = self.cam1.read()
            ret2, frame2 = self.cam2.read()

            if ret1 and ret2 is True:
                return frame1, frame2

        def find_ball(frame1, frame2):
            ball_pos1 = bt.find_ball(frame1, 1)
            ball_pos2 = bt.find_ball(frame2, 2)

            return ball_pos1, ball_pos2

        def generate_3d_point(pos1, pos2):
            # Generates 3d points from 2 2D points, duh
            if is_zero(pos1) or is_zero(pos2):
                return np.array([0, 0, 1])
            else:
                point = self.calc_3d_point(pos1, pos2)
                if inside_range(point):
                    return point
                else:
                    return np.array([0, 0, 0])

        def remove_point_outliers():
            # Remove outliers in ball points, duhhhh
            samecount = 0
            for i in range(np.size(self.p3d, 0) - 4):
                neighs = []
                for j in range(5):
                    if not is_zero(self.p3d[i + j, :]) and j != 2:
                        neighs.append(self.p3d[i + j, :])
                if len(neighs) > 0:
                    arr = np.array(neighs)
                    means = np.mean(arr, axis=0)
                    norm = np.linalg.norm(self.p3d[i + 2, :] - means)
                    if norm > 0.5 and samecount < 10:
                        samecount += 1
                        self.p3d[i + 2, :] = 0
                    else:
                        samecount = 0

        # Ze master function!!!
        def update_points(i):

            # Grab Frames from cap
            frame1, frame2 = grab_frames()

            # Then, find ball in both frames (2D)
            ball_pos1, ball_pos2 = find_ball(frame1, frame2)

            # Then, from those 2D points find 3D pos based on math
            ball_3d_pos = generate_3d_point(ball_pos1, ball_pos2)

            # Add to rolling buffer of data points
            self.p3d.append(ball_3d_pos)

            # Tweak data and remove outliers
            #remove_point_outliers()

            # Update graphically
            ax.clear()  # Because resetting the graph removes old balls
            plot_table(ax)
            plot_cameras(ax)

            # Add all points
            for i in range(len(self.p3d)):
                curPoint = self.p3d[i - 1]
                ax.scatter(curPoint[0], curPoint[1], curPoint[2])

        # Starts cam / tracking / animation loop based on interval!
        anim = animation.FuncAnimation(fig, update_points, interval=interval)
        plt.show()


# ----- Special Functions -----

# RQ-factorization
def rq(a):
    '''
    :param a:       Original matrix
    :return: r,q    rq=a
    '''

    [m, n] = a.shape
    e = np.eye(m)
    p = np.fliplr(e)
    [q0, r0] = qr(np.matmul(p, np.matmul(np.transpose(a[:, 0:m]), p)))
    r = np.matmul(p, np.matmul(np.transpose(r0), p))
    q = np.matmul(p, np.matmul(np.transpose(q0), p))
    fix = np.diag(np.sign(np.diag(r)))
    r = np.matmul(r, fix)
    q = np.matmul(fix, q)
    if n > m:
        q = np.concatenate((q, np.matmul(np.linalg.inv(r), a[:, m:n])), axis=1)
    return r, q


# Pointwise division with last coordinate
def pflat(x):
    y = np.copy(x)
    for i in range(x.shape[1]):
        y[:, i] = y[:, i] / y[x.shape[0] - 1, i]
    return y


# Calculates camera matrix from a set of 6 point correspondences
def calc_P(p3d, p2d):
    '''
    :param p3d:     3D known points
    :param p2d:     Corresponding 2D points in cameras
    :return:        Camera matrix
    '''
    npoints = p2d.shape[1]
    mean = np.mean(p2d, 1)
    std = np.std(p2d, axis=1)
    N = np.array([[1 / std[0], 0, -mean[0] / std[0]],
                  [0, 1 / std[1], -mean[1] / std[1]],
                  [0, 0, 1]])
    p2dnorm = np.matmul(N, p2d)
    M = np.zeros([3 * npoints, 12 + npoints])
    for i in range(npoints):
        M[3 * i, 0:4] = p3d[:, i]
        M[3 * i + 1, 4:8] = p3d[:, i]
        M[3 * i + 2, 8:12] = p3d[:, i]
        M[3 * i:3 * i + 3, 12 + i] = -p2dnorm[:, i]
    [U, S, V] = svd(M)
    v = V[V.shape[0] - 1, :]
    P = np.reshape(v[0:12], [3, 4])
    testsign = np.matmul(P, p3d[:, 1])
    if testsign[2] < 0:
        P = -P
        print('changed sign of P')
    P = np.matmul(np.linalg.inv(N), P)
    return P


# Checks if point is zero and should be ignored
def is_zero(p):
    '''
    :param p:   Point
    :return:    true if both coordinates are zero
    '''
    if p[0] == 0 and p[1] == 0:
        return True
    else:
        return False


# Checks if point is within reasonable range from table
def inside_range(point):
    '''
    :param point:   Detected points
    :return:        Indexes of point positions inside a range
    '''
    return -1 < point[0] < 3.74 and -1 < point[1] < 2.525 and -1 < point[2] < 3


# Interpolate positions of missing points, bicubic interpolation
def interpolate_missing(a, b, c, d, t0, t1, t2, t3):
    '''
    Takes four points with corresponding weight for bicubic interpolation
    :param a:
    :param b:
    :param c:
    :param d:
    :param t0:
    :param t1:
    :param t2:
    :param t3:
    :return:
    '''
    matinv = np.linalg.inv([[1, t0, t0 ** 2, t0 ** 3],
                            [1, t1, t1 ** 2, t1 ** 3],
                            [1, t2, t2 ** 2, t2 ** 3],
                            [1, t3, t3 ** 2, t3 ** 3]])
    coeff = np.zeros([3, 4])
    for i in range(3):
        values = np.array([a[i], b[i], c[i], d[i]])
        coeff[i, :] = matinv @ values
    missing = np.zeros([t3 - t0 + 1, 3])
    for i in range(missing.shape[0]):
        missing[i, :] = coeff @ np.array([1, t0 + i, (t0 + i) ** 2, (t0 + i) ** 3])
    return missing