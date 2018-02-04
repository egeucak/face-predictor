from keras.models import Sequential

from .network import Network
from .tools import Tools

class Point_Predictor:
    def __init__(self):
        self.master_tool = Tools()
        model = Sequential()
        temp = Network(model, no=1, predict=True)
        self.model = temp.get_model()

        for layer in self.model.layers:
            layer.trainable = False

        self.model.compile(optimizer="adagrad",
                      loss="mean_squared_error",
                      metrics=["accuracy"])

        self.X, _ = self.master_tool.load(test=True, fill=False)
        print(self.X.shape)
        self.Y = model.predict(self.X)

    def predict_points(self, face):
        try:
            X = face
            print(X.shape)
            Y = self.model.predict(X)
            self.master_tool.plot_sample(X[0], Y[0])
            self.master_tool.plot_sample(self.X[0], self.Y[0])

        except Exception as e:
            print("An error occured...")
            print(e)