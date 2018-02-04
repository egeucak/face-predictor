from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential

import keras

from src.tools import Tools
from src.network import Network

from src.face_detector import Face_Detector
from src.face_point_predictor import Point_Predictor

class Visualize(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        detector = Face_Detector()
        self.img = detector.detect("pic2.jpg")
        print(self.img)

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        result = self.model.predict(self.img)
        Tools.plot_sample(self.img, result)

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def create_relu_advanced(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced

class Test(keras.callbacks.Callback):
    def __init__(self):
        return

master_tool = Tools()


X, y = master_tool.load(fill=False)

print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))

earl = EarlyStopping(monitor="val_loss",
                     min_delta=0,
                     patience=10)

model = Sequential()
temp = Network(model=model, no=3)
model = temp.get_model()

#model.load_weights("model.h5")

model.compile(optimizer="adagrad",
              loss="mean_squared_error",
              metrics=["accuracy"])

model.fit(X, y,
          callbacks=[Test],
          epochs=100,
          verbose=1,
          validation_split=0.1,
          shuffle=True
          )

model.save("model4.h5")
