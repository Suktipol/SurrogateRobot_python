import numpy as np
import cv2, PIL, os
from cv2 import aruco
import socket

UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 8053
clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  #Create client to send data to server

##############################################################################################
#############################     read_chessboards     #######################################
##############################################################################################
def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

##############################################################################################
#############################     calibrate_camera     #######################################
##############################################################################################
def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


##############################################################################################
#############################      Main Function      ########################################
##############################################################################################

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
imboard = board.draw((2000, 2000))
cv2.imwrite("chessboard.jpg", imboard)

datadir = "Charuco_pic/"
images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".jpg") ])
order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
images = images[order]
#im = PIL.Image.open(images[0])

allCorners, allIds, imsize = read_chessboards(images)

ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize)

#############################   Finish Calibration   ########################################

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

while (True):
    ret, frame = cap.read()
    imaxis = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
                                                          parameters=parameters)
    # SUB PIXEL DETECTION
    for corner in corners:
        cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, 1), criteria=criteria)

    if np.all(ids != None):
        frame_markers = aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, mtx, dist)
        b = bytes()
        if (ids.size == 1):
            rvec_x = int(rvecs[0][0][0] * 100) / 100
            rvec_y = int(rvecs[0][0][1] * 100) / 100
            rvec_z = int(rvecs[0][0][2] * 100) / 100
            tvec_x = int(tvecs[0][0][0] * 100) / 100
            tvec_y = int(tvecs[0][0][1] * 100) / 100
            tvec_z = int(tvecs[0][0][2] * 100) / 100
            r_t_vecs = str(ids[0][0]) + ',' + \
                       str(rvec_x) + ',' + \
                       str(rvec_y) + ',' + \
                       str(rvec_z) + ',' + \
                       str(tvec_x) + ',' + \
                       str(tvec_y) + ',' + \
                       str(tvec_z)

            Message = bytes(r_t_vecs, 'utf-8')  # convert string to bytes
            clientSock.sendto(Message, (UDP_IP_ADDRESS, UDP_PORT_NO))  # Send data

        imaxis = aruco.drawAxis(frame, mtx, dist, rvecs[0], tvecs[0], 0.05)

    #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow('frame', imaxis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


