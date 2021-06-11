import cv2
import numpy as np
from numpy.core.numeric import full
np.set_printoptions(threshold=np.inf)
HOST = '192.168.1.104'  # Standard loopback interface address (localhost)
PORT = 65433
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))
full_msg = b''
anz_sch = 0
while True:
    anz_sch += 1
    msg = s.recv(1024*100)
    if len(msg)<=0:
        break        
    #print(msg)
    full_msg += msg#.decode("utf-8")
    suche = full_msg.find(b'#-----#')    
       
    if(suche) != -1:
        print("Lange von full_msg", len(full_msg))
        print("suche:  ", suche)
        print(anz_sch)
        nparr = np.fromstring(full_msg[:suche], np.uint8)#[-1000:]
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

        print(img_np.shape)
        img_res = cv2.resize(img_np, (1920, 960))                    # Resize image
        cv2.imshow("Uebertragenes Bild", img_res)
        cv2.waitKey(1)
        print("nach cv2")        
        full_msg = full_msg[suche+7:]
