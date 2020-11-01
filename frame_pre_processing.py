import numpy as np
import cv2
import subprocess


# load the anti-distorsion matrix
npz_calib_file = np.load('resources/calibration_data.npz')
distCoeff = npz_calib_file['distCoeff']
intrinsic_matrix = npz_calib_file['intrinsic_matrix']


def pre_process(frame, undistort=False):

    # perform anti-distorsion
    if undistort:
        frame = cv2.undistort(frame, intrinsic_matrix, distCoeff, None)

       
    # focus measure of the image using the Variance of Laplacian method
    # https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()


    # for the worst frame we use a SRN-Deblur Network to perform motion deblurring
    # https://github.com/jiangsutx/SRN-Deblur
    if focus_measure <= 35:
        cv2.imwrite('SRN-Deblur/testing_set/' + 'tmp' + '.png', frame)
        p = subprocess.Popen(['env/bin/python', 'run_model.py', '--input_path=./testing_set', '--output_path=./testing_res'], cwd='./SRN-Deblur')
        p.wait()
        frame = cv2.imread('SRN-Deblur/testing_res/' + 'tmp' + '.png')


    return frame