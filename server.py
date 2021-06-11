import socket
import cv2
HOST = '192.168.1.104'  # Standard loopback interface address (localhost)
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
        frame = cv2.resize(frame, (2024,1080))

        print(frame.shape)
        cv2.imshow('uebertragung', frame)
        cv2.waitKey(1)
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        bytes = im_buf_arr.tobytes()
        #clientsocket.recv(1024*10)
        print(f"Verbindung von {address} wurde hergestellt!")
        clientsocket.sendall((bytes))
        clientsocket.sendall((b'#-----#'))

        print("Ende von while")
        #clientsocket.close()




'''
import socket
import cv2
import pickle
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('Host IP: ', host_ip)
port = 1234
socket_address = (host_ip, port)
server_socket.bind(socket_address)

server_socket.listen(5)
print("Listening at", socket_address)

while True:
    clientsocket, address = server_socket.accept()
    print(f"Verbindung von {address} wurde hergestellt!")
    clientsocket.send(bytes("Welcome to the server", "utf-8"))
'''
'''
    client_socket, addr = server_socket.accept()
    print("got connection vrom", addr)
    if client_socket:
        vid = cv2.VideoCapture(0)
        while(vid.isOpened()):
            img, frame = vid.read()
            a = pickle.dumps(frame)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)
            cv2.imshow('uebertragung', frame)
            cv2.waitKey(1)
    '''       
''' 
import socket

HOST = '192.168.1.104'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            #if not data:
            #    break
            conn.sendall(bytes("penis", "utf-8"))
            #conn.close()
'''