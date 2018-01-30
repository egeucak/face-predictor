from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import os
import numpy as np
import math
import matplotlib.pyplot as plt

class Tools:
    def __init__(self):
        self.FTRAIN = "training.csv"
        self.FTEST = "test.csv"

    def load(self, test=False, cols=None, fill=True):
        fname = self.FTEST if test else self.FTRAIN
        df = read_csv(os.path.expanduser(fname))

        df["Image"] = df["Image"].apply(lambda im: np.fromstring(im, sep=" "))

        if cols:
            df = df[list(cols) + ["Image"]]

        # print(df.count())
        df = df.dropna()

        X = np.vstack(df["Image"].values) / 255.
        X = X.astype(np.float32)

        if not test:
            y = df[df.columns[:-1]].values
            y = y / 96
            X, y = shuffle(X, y, random_state=42)
            y = y.astype(np.float32)

            if fill:
                check_agains = []
                means = [[] for i in range(30)]
                for index, face in enumerate(y):
                    [means[index2].append(point) for index2, point in enumerate(face) if math.isnan(point) == False]
                    # if "nan" in face: check_agains.append(index)
                    if np.sometrue(np.isnan(face)): check_agains.append(index)

                means = [np.mean(num) for index3, num in enumerate(means)]
                for failed in check_agains:
                    for loc, val in enumerate(y[failed]):
                        if math.isnan(val) == False: continue
                        y[failed][loc] = means[loc]
        else:
            y = None

        return X.reshape(-1, 96, 96, 1), y

    def plot_sample(self, x, y):
        plt.figure()
        img = x.reshape(96, 96)
        plt.imshow(img, cmap="gray")
        plt.scatter(y[0::2] * 96, y[1::2] * 96, marker="x")
        plt.show()

    def decode_locations(self, y):
        print("Left eye center => {} - {}".format(y[0], y[1]))
        print("Right eye center => {} - {}".format(y[2], y[3]))
        print("Left eye inner corner => {} - {}".format(y[4], y[5]))
        print("Left eye outer corner => {} - {}".format(y[6], y[7]))
        print("Right eye inner corner => {} - {}".format(y[8], y[9]))
        print("Right eye outer corner => {} - {}".format(y[10], y[11]))
        print("Left eyebrow inner-end => {} - {}".format(y[12], y[13]))
        print("Left eyebrow outer-end => {} - {}".format(y[14], y[15]))
        print("Right eyebrow inner-end => {} - {}".format(y[16], y[17]))
        print("Right eyebrow outer-end => {} - {}".format(y[18], y[19]))
        print("Nose tip => {} - {}".format(y[20], y[21]))
        print("Mouth left corner => {} - {}".format(y[22], y[23]))
        print("Mouth right corner => {} - {}".format(y[24], y[25]))
        print("Mouth center top lip => {} - {}".format(y[26], y[27]))
        print("Mouth center bottom lip => {} - {}".format(y[28], y[29]))