import numpy as np
import cv2
from resizeimage import resizeimage
from PIL import Image

class Face_Detector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def detect(self, loc):
        img = cv2.imread(loc)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_gray = gray[y: y+h, x:x+w]
            roi_color = img[y: y+h, x:x+w]

        roi_gray = Image.fromarray(roi_gray)
        roi_gray = resizeimage.resize_thumbnail(roi_gray, [96,96])
        roi_gray = np.array(roi_gray).reshape((1,96,96,1))
        # cv2.imshow("img", roi_gray)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return roi_gray/255.