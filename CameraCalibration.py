import numpy as np
import glob
import cv2

def init():
    global ret, mtx, dist, rvecs, tvecs

images = glob.glob('*.jpg')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1, 2)
objPoints = []
imgPoints = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #dimention of the real chessboard ==> (9,7)
    #dimention in chessboard function ==> minus 1
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    if ret == True:
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
cv2.destroyAllWindows()

