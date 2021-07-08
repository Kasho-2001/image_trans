import argparse
import time
from pathlib import Path
import base64
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class cv_detection:
    def __init__(self):
        self.device = ""
        self.device = select_device(self.device)
        self.model = attempt_load('yolov5s.pt', map_location=self.device)  # load FP32 model
        print("Modell geladen")
        self.imageSize = 1024
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA
        # self.rospack = rospkg.RosPack()
        # self.rospath = rospack.get_path('cv_xi_21')

        self.imageSize = check_img_size(
            self.imageSize, s=self.model.stride.max()
        )  # check img_size
        if self.half:
            self.model.half()  # to FP16

        self.saveConfiguration = False
        self.saveImage = True
        self.viewImage = False

        self.confidenceThreshold = 0.45
        self.iouThreshold = 0.45
        self.names = self.model.module.names if hasattr(self.model, "module") else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect(self, image, fps="was geeeeeht?"):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Bei Matrix Vision Kameras war ne Blau Rot Verwechselung 
        # Initialize
        set_logging()
        # Set Dataloader




        dataset = LoadImages(
            image, img_size=self.imageSize
        )  # Datentyp von dataset anschauen

        # Run inference
        t0 = time.time()
        img = torch.zeros(
            (1, 3, self.imageSize, self.imageSize), device=self.device
        )  # init img
        # run once
        _ = (
            self.model(img.half() if self.half else img)
            if self.device.type != "cpu"
            else None
        )  # "_ ist Konvention"
       

        for (
            path,
            img,
            im0s,
            vid_cap,
            img_width,
            img_height,
        ) in dataset:  # Todo (For Schleife wird wahrscheinlich nicht benötigt)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment='')[0]
           
            classes= 0   #   =None wenn alle Klassen angezeigt werden sollen
            agnostic = False
            # Apply NMS
            pred = non_max_suppression(
                pred,
                self.confidenceThreshold,
                self.iouThreshold,
                classes=classes,
                agnostic=agnostic,
            )
            t2 = time_synchronized()
            boundingBoxListe = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image#
                if len(det):  # Anschauen was det genau is

                    boundingBoxListe = self.processDetection(
                        im0s, dataset, img, det, img_width, img_height, vid_cap
                    )
            #im0s = cv2.resize(im0s, (1800, 950))                    # Resize image
            #print("fps", fps)
            #cv2.putText(im0s, fps, (50,80), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5)

            #cv2.imshow("f1", im0s)
            #cv2.waitKey(1)
            #print(f"{boundingBoxList}Done. ({t2 - t1:.3f}s)")

            return boundingBoxListe
        return        

    def processDetection(self, im0, dataset, img, det, img_width, img_height, vid_cap):
        frame = getattr(dataset, "frame", 0)
       

        # normalization gain whwh
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        return self.writeResult(
            det,
            gn,
            img_width,
            img_height,
            im0,
    
        )

    def writeResult(self, det, gn, image_width, image_height, im0):
        # Get names and colors
        boundingboxliste = []
        for *xyxy, conf, cls in reversed(det):
            label = f'{self.names[int(cls)]} {conf:.2f}' #nur fürs Testen benötigt
            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3) #nur fürs Testen benötigt(Damit BB buntisch angezeigt werden)
            boundingbox = []


            label = f"{self.names[int(cls)]} {int(cls):.2f}"
            xywh = (
                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            )  # normalized xywh
            
            boundingbox.append(int(xyxy[0]))
            boundingbox.append(int(xyxy[1]))
            boundingbox.append(int(xyxy[2]))
            boundingbox.append(int(xyxy[3]))
            boundingbox.append(float(f"{conf:.2f}"))
            boundingboxliste.append(boundingbox)
        return boundingboxliste