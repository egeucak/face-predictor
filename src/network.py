from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Activation
from keras. layers import Convolution2D, MaxPooling2D, AveragePooling2D

class Network:
    def __init__(self, model, no=0, predict=False):
        if no == 0:       # model-weird.h5
            model.add(Convolution2D(filters=32, kernel_size=(2,2), activation="relu", input_shape=(96, 96, 1)))
            model.add(Convolution2D(filters=48, kernel_size=(2,2), activation="relu"))
            model.add(MaxPooling2D())
            model.add(Convolution2D(filters=64, kernel_size=(3,3), activation="relu"))
            model.add(Convolution2D(filters=96, kernel_size=(3,3), activation="relu"))
            model.add(MaxPooling2D())
            model.add(Flatten())
            model.add(Dense(64, activation="relu"))
            model.add(Dense(30, activation="relu"))
            if predict: model.load_weights("../models/model-weird.h5")
            self.model = model

        elif no == 1:     # model2.h5
            model.add(Convolution2D(filters=32, kernel_size=(2, 2), input_shape=(96, 96, 1)))
            model.add(LeakyReLU())

            model.add(Convolution2D(filters=48, kernel_size=(2, 2)))
            model.add(LeakyReLU())

            model.add(MaxPooling2D())

            model.add(Convolution2D(filters=64, kernel_size=(3, 3)))
            model.add(LeakyReLU())

            model.add(Convolution2D(filters=96, kernel_size=(3, 3)))
            model.add(LeakyReLU())

            model.add(MaxPooling2D())

            model.add(Flatten())
            model.add(Dense(64))
            model.add(LeakyReLU())
            model.add(Dense(30, activation="linear"))
            if predict: model.load_weights("../models/model2.h5")
            self.model = model

        elif no == 3:
            model.add(Convolution2D(filters=32, kernel_size=(2, 2), input_shape=(96, 96, 1)))
            model.add(LeakyReLU(alpha=0.1))

            model.add(Convolution2D(filters=40, kernel_size=(2, 2)))
            model.add(LeakyReLU(alpha=0.1))

            model.add(MaxPooling2D())

            model.add(Convolution2D(filters=48, kernel_size=(3, 3), strides=2))
            model.add(LeakyReLU(alpha=0.1))

            model.add(Convolution2D(filters=56, kernel_size=(3, 3), strides=2))
            model.add(LeakyReLU(alpha=0.1))

            model.add(MaxPooling2D())

            model.add(Flatten())
            model.add(Dropout(0.5))
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(30, activation="linear"))
            if predict: model.load_weights("../models/model3.h5")
            self.model = model


    def get_model(self):
        return self.model