import numpy as np
from cv2 import aruco
import cv2
import CameraCalibration as CamC
import socket

###
## Create Marker
###
'''markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
img = np.zeros((400, 400), dtype=np.uint8)
aruco.drawMarker(markerDict, 4, 400, img, 1)'''

#for TCP communication
TCP_IP_ADDRESS = "127.0.0.1"
TCP_PORT_NO = 8053
tcpSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpSock.connect((TCP_IP_ADDRESS, TCP_PORT_NO))

cap = cv2.VideoCapture(0)

CamC.init()

markerDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

status = 2
old_status = 2

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame  #set default of img
    corners, ids, rejectedCandidate = aruco.detectMarkers(gray, markerDict, parameters=parameters)

    if np.all(ids != None):
        if (ids.size == 1):
            img = aruco.drawDetectedMarkers(frame, corners, ids, (255, 0, 0))
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.08, CamC.mtx, CamC.dist)
            tvec_z = int(tvecs[0][0][2] * 100) / 100  #Check distance between camera & marker
            if (tvec_z <= 2.5):
                status = 1      #Marker is detected
            else:
                status = 0      #Marker is undetected
    else:
        status = 0

    if (status != old_status):
        if (ids != None):
            text = str(status) + ',' + str(ids[0][0])
        else:
            text = str(status) + ',' + str(ids)

        byteMessage = bytes(text, 'utf-8')
        tcpSock.sendall(byteMessage)
        old_status = status

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
tcpSock.close()
cv2.destroyAllWindows()