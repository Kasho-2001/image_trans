import socket
import cv2
import numpy as np
from numpy.core.numeric import full
import detect_person
import time

np.set_printoptions(threshold=np.inf)
HOST = '192.168.1.110'  # Standard loopback interface address (localhost)
PORT = 65433
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST,PORT))
full_msg = b''
anz_sch = 0
afps =[]


dp = detect_person.cv_detection()

startbildempf = 0
startyolo= 0
nachyolo = 0


def det(img_np, fps):
    dp.detect(img_np, fps)



while True:
    anz_sch += 1
    msg = s.recv(1024*10000)
    if len(msg)<=0:
        break        
    #print(msg)
    full_msg += msg#.decode("utf-8")
    ende = full_msg.rfind(b'#---#')    
    
    
    if(ende) != -1:
        print("Dl Zeit", time.time()-startbildempf)
        #print("Lange von full_msg", len(full_msg))
        #print("suche:  ", suche)
        #print(anz_sch)
        img_msg = full_msg[:ende]

        anfang = img_msg.rfind(b'--#--')

        nparr = np.frombuffer(img_msg[anfang+5:], np.uint8)#[-1000:] #davor fromstring
        #print("np", full_msg[anfang+5:])
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        img_np = cv2.resize(img_np, (2000, 950))                    # Resize image
        fps = 1 / (time.time() - startyolo)
        #afps.append(fps)
        #dfps = sum(afps)/len(afps)
        #print(fps)
        print("\Yolozeit: ", time.time()-nachyolo)
        startyolo = time.time()
        det(img_np, str(round(fps,2)))
        nachyolo = time.time()
        print("YoloZeit:", nachyolo-startyolo)
        #cv2.imshow("Uebertragenes Bild", img_np)
        #cv2.waitKey(1)

        #print(img_np.shape)
       
        #print("nach cv2")        
        full_msg = full_msg[ende+5:]
        anz_sch = 0
        startbildempf = time.time()
        #Als nächstes eventuell multiprocessing oder ans Zählen gehen!
