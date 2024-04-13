import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2 as cv

def load(img):
    out = io.imread(img)
    return out

def display(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def calibrate():
    image_filenames = [ 'IMG (1).jpg',
                        'IMG (2).jpg',
                        'IMG (3).jpg',
                        'IMG (4).jpg',
                        'IMG (5).jpg']
    
    rows = 6
    columns = 10
    terminationCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((rows*columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    
    for filename in image_filenames:
        img = load(filename)
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        ret, corners = cv.findChessboardCorners(gray, (columns, rows), None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), terminationCriteria)
            imgpoints.append(corners2)
            
            img = cv.drawChessboardCorners(img, (columns, rows), corners2, ret)
            display(img)
            cv.waitKey(500)

    cv.destroyAllWindows()

    return cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)


def undistort(img, mtx, dist):
    imgUndistorted = cv.undistort(img, mtx, dist, None, mtx)
    return imgUndistorted