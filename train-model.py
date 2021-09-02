import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

proj_dir = os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(proj_dir, "images")
image_train_dir = os.path.join(image_dir, "train")

cascade = cv2.CascadeClassifier('../Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
#cascade = cv2.CascadeClassifier('../Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

labels=[]
train_samples=[]

redcolor = (255, 0, 0) #red
greencolor = (0, 255, 0) #greed
stroke = 2

for root, dirs, files in os.walk(image_train_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root,file)

            image = cv2.imread(path) 
            
            img = Image.open(path).convert("L")
            img_arr = np.array(img, "uint8")
            faces = cascade.detectMultiScale( img_arr, 1.1, 2, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(30, 30) );
            for (x,y,w,h) in faces:
                #print("found a face")
                roi=img_arr[y:y+h, x:x+w]
                train_samples.append(roi)
                labels.append(0)

                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(image, (x, y), (end_cord_x, end_cord_y), redcolor, stroke)
            
            # Display frame
            plt.imshow(image)
            plt.show()
            

recognizer.train(train_samples, np.array(labels))
recognizer.save("model/hoffman-model.yml")