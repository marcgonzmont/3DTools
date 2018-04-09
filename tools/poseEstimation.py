import cv2
import numpy as np
from utilities.utilities import plotImages as plI

def drawAxis(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def compute(mtx, dist, all_images, plt_axis):
    # Pattern size
    size = (7, 6)
    axis_len = 2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((size[1] * size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0: size[0], 0: size[1]].T.reshape(-1, 2)

    if plt_axis:
        axis = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]]).reshape(-1, 3)
    else:

        axis = np.float32([[0, 0, 0], [0, axis_len, 0], [axis_len, axis_len, 0], [axis_len, 0, 0],
                           [0, 0, -axis_len], [0, axis_len, -axis_len], [axis_len, axis_len, -axis_len], [axis_len, 0, -axis_len]]).reshape(-1, 3)

    for fname in all_images:
        img = cv2.imread(fname)
        cp_img = img.copy()
        gray = cv2.cvtColor(cp_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            if plt_axis:
                cp_img = drawAxis(cp_img, corners2, imgpts)
            else:
                cp_img = drawCube(cp_img, corners2, imgpts)

            title = 'Pose estimation'
            titles = ['Calibrated', 'With axis']
            images = [img, cp_img]
            plI(titles, images, title, 1, 2)
