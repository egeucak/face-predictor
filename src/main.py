import time

from src.face_detector import Face_Detector
from src.face_point_predictor import Point_Predictor

detector = Face_Detector()
point_predictor = Point_Predictor()


def get_face(loc):
    img = detector.detect(loc)
    point_predictor.predict_points(img)


start = time.time()
get_face("pic.jpg")
get_face("pic2.jpg")
get_face("pic3.jpg")
print("3 results in {} ms".format(time.time() - start))
