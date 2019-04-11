# Again we import the necessary socket python module
import socket
import numpy as np
import cv2

UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 8052

frame = []
frame = np.array(frame, dtype=np.uint8)
i = 0

serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSock.bind((UDP_IP_ADDRESS, UDP_PORT_NO))
#serverSock.sendto(b'/x/x/x/x/x/', (UDP_IP_ADDRESS, UDP_PORT_NO))
while True:
    data, addr = serverSock.recvfrom(64800)
    data = np.frombuffer(data, np.uint8, len(data), 0)
    frame = np.concatenate((frame, data))
    if ( len(frame) == 1555200 ):
        frame = np.array(frame, dtype=np.uint8)
        frame = np.reshape(frame, (540,960,3))
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cv2.imshow("Hello C#", gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame = []



cv2.destroyAllWindows()
