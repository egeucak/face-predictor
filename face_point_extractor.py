from keras.models import Sequential

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential

from tools import Tools
from network import Network

def create_relu_advanced(max_value=1.):
    def relu_advanced(x):
        return K.relu(x, max_value=K.cast_to_floatx(max_value))
    return relu_advanced

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
          epochs=100,
          verbose=1,
          validation_split=0.1,
          shuffle=True,
          callbacks=[])

model.save("model4.h5")
