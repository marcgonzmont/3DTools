import cv2
import numpy as np
from time import time
from utilities.utilities import plotImages as plI


def calculateError(objpoints, imgpoints, mtx, dist, rvecs, tvecs):

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("Total calibration error: {:2.4f}".format(mean_error / len(objpoints)))


def calibrate(template, image, adv, plot):
    start = time()
    # Pattern size
    size = (7, 6)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((size[1] * size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : size[0], 0 : size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    img_tmpl = cv2.imread(template)
    img_test = cv2.imread(image)
    gray = cv2.cvtColor(img_tmpl, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img_tmpl, size, corners, ret)
        # cv2.imshow('Template', img)
        # cv2.waitKey(1000)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if adv:
        h, w = img_test.shape[:2]
        newcameramtx = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))[0]
        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img_test, mapx, mapy, cv2.INTER_LINEAR)

    else:
        # undistort
        # dst = cv2.undistort(img_test, mtx, dist, None, newcameramtx)
        dst = cv2.undistort(img_test, mtx, dist)
    print("Camera calibration takes: {:2.5f} seconds".format(time()-start))
    calculateError(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    if plot:
        print("ESTIMATED PARAMS\n"
              "Ret:\n{}\n\n"
              "Mtx:\n{}\n\n"
              "Dist:\n{}\n\n"
              .format(ret, mtx, dist))

        title = 'Camera calibration'
        titles = ['Template', 'Calibrated']
        images = [img_tmpl, dst]
        plI(titles, images, title, 1, 2)

    return mtx, dist
