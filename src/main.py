import numpy as np
import cv2
from resizeimage import resizeimage
from PIL import Image

from src.face_detector import Face_Detector
from src.face_point_predictor import Point_Predictor

detector = Face_Detector()
point_predictor = Point_Predictor()

img = detector.detect("pic.jpg")
img2 = detector.detect("pic2.jpg")

point_predictor.predict_points(img)
point_predictor.predict_points(img2)