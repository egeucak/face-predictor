import time

from src.face_detector import Face_Detector
from src.face_point_predictor import Point_Predictor
from src.Person import Face

detector = Face_Detector()
point_predictor = Point_Predictor()


def get_face(loc):
    img = detector.detect(loc)
    return point_predictor.predict_points(img)

ege = Face(loc="ege", load=True)
# ege.add(get_face("pic4.jpg"))
# ege.add(get_face("pic.jpg"))
# ege.add(get_face("pic2.jpg"))

# ege.save("ege")

print(ege.check(get_face("pic4.jpg")))
