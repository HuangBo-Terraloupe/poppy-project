import socket
import pickle
import cv2
import numpy as np



if __name__ == '__main__':

    camera = cv2.VideoCapture(1)
    TCP_IP = '129.187.105.109'
    TCP_PORT = 20000
    BUFFER_SIZE = 1024
    # MESSAGE = "Hello, World!"

    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s = socket.socket()
    s.connect((TCP_IP, TCP_PORT))

    while True:
        string = raw_input('position: x,y\n')
        string = string.split(',')
        x = float(string[0])
        y = float(string[1])

        while True:
            s.send(pickle.dumps(np.array([x,y])))
            if pickle.loads(s.recv(BUFFER_SIZE)):
                break

    s.close()
