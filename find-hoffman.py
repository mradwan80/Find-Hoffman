import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

cascade = cv2.CascadeClassifier('../Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
#cascade = cv2.CascadeClassifier('../Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./model/hoffman-model.yml")


proj_dir = os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(proj_dir, "images")
image_test_dir = os.path.join(image_dir, "test")

red_color = (255, 0, 0)
green_color = (0, 255, 0)
stroke = 2

for root, dirs, files in os.walk(image_test_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):

            path = os.path.join(root,file)
            
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = cascade.detectMultiScale( gray, 1.1, 2, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30) );
            
            minconf=200
            for (startx, starty, w, h) in detected_faces:
                roi_gray = gray[starty:starty+h, startx:startx+w]
                roi_color = img[starty:starty+h, startx:startx+w]

                
                endx = startx + w
                endy = starty + h

                id_, conf = recognizer.predict(roi_gray)
                if conf <= 150 and conf<minconf:
                    minconf=conf
                    hof_startx=startx
                    hof_starty=starty
                    hof_endx=endx
                    hof_endy=endy
                
                cv2.rectangle(img, (startx, starty), (endx, endy), red_color, stroke)
                    

            if minconf<200:
                cv2.rectangle(img, (hof_startx, hof_starty), (hof_endx, hof_endy), green_color, stroke)
                

            # Display frame
            plt.imshow(img)
            plt.show()
            

cv2.destroyAllWindows()
