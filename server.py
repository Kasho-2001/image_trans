import socket
import cv2
HOST = '192.168.1.110' #auf wlan Einstellungen -->Ipv4 Adresse
PORT = 65433
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(5)  #Mögliche Klienten in der Warteschlange


#bytes = open("test.png", 'rb').read()
#print(bytes)





vid = cv2.VideoCapture(0)
while True:        
    clientsocket, address = s.accept()

    while(vid.isOpened()):
        print("Anfang von while")
        #clientsocket.sendall((bytes))
        img, frame = vid.read()
        frame = cv2.resize(frame, (2096,1450))

        print(frame.shape)
        cv2.imshow('uebertragung', frame)
        cv2.waitKey(1)
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        bytes = im_buf_arr.tobytes()
        #clientsocket.recv(1024*10000)
        print(f"Verbindung von {address} wurde hergestellt!")

        clientsocket.sendall((b'--#--'))
        clientsocket.sendall((bytes))
        clientsocket.sendall((b'#---#'))

        print("Ende von while")
        #clientsocket.close()