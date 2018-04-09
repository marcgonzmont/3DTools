import argparse
from os.path import basename
import numpy as np
import random
from utilities import utilities as util
from tools import cameraCalibration as calib
from tools import poseEstimation as posEst

if __name__ == '__main__':
    # para cada uno de los modos incluir grupos de argumentos de forma excluyente
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True,
                        help="-d 3D coordinates of the points (real world)")
    parser.add_argument("-p", "--path", required=True,
                    help="-p Path where the template and images are stored")
    parser.add_argument("-m", "--mode", required=False, default= 0,
                        help="-m Mode for undistorsion (0: cv2.undistort(), 1: cv2.initUndistortRectifyMap())")
    args = vars(parser.parse_args())

    # Configuration
    file_ext = '.jpg'
    template_idx = 0
    axis = 1
    plot = 1
    pose = 0
    # tmp_substr = '*Plantilla*'
    # usr_substr = '*Individuo*'

    # cada una de las imagenes de la práctica corresponde a una calibración: template1 --> user1 and so on
    # Load images and data
    all_images = util.getSamples(args["path"], file_ext)
    template = all_images[template_idx]
    del all_images[template_idx]
    image = random.choice(all_images)
    print("\n--------------------------\n"
          "CAMERA CALIBRATION\n"
          "Template: {}\n"
          "Test image: {}\n"
          "Mode: {}".format(basename(template), basename(image), args["mode"]))

    mtx, dist = calib.calibrate(template, image, int(args["mode"]), plot)
    if pose:
        posEst.compute(mtx, dist, all_images, axis)

