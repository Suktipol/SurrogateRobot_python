import socket

UDP_IP_ADDRESS = "10.61.5.13"
UDP_PORT_NO = 8052
Message = bytes(DM.b, 'utf-8')

clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

clientSock.sendto(Message, (UDP_IP_ADDRESS, UDP_PORT_NO))