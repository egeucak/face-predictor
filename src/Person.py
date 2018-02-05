import numpy as np

class Face:
    def __init__(self, load=False, loc=None):
        if load or loc:
            self.points = np.load("../faces/{}.npy".format(loc))
        else:
            self.points = []

    def add(self, dot):
        self.points.append(dot)

    def check(self, face):
        diff = []
        face = np.asarray(face)
        for point in self.points:
            for P1, P2 in zip(face, point):
                diff.append(P1**2 - P2**2)
        return np.sum(diff) / len(diff)

    def save(self, name):
        try:
            np.save(file="../faces/{}.npy".format(name), arr=self.points)
            return True
        except Exception as e:
            return False