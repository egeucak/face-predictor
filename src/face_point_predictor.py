from keras.models import Sequential

from .network import Network
from .tools import Tools

master_tool = Tools()

model = Sequential()
temp = Network(model, no=3, predict=True)
model = temp.get_model()

for layer in model.layers:
    layer.trainable = False

model.compile(optimizer="adagrad",
              loss="mean_squared_error",
              metrics=["accuracy"])

X, _ = master_tool.load(test=True, fill=False)
Y = model.predict(X)

while 1:
    try:
        number = input("Please enter a number...\n>>>")
        if number == "e": exit()
        number = int(number)
        # master_tool.decode_locations(Y[number])
        master_tool.plot_sample(X[number], Y[number])

    except Exception as e:
        print("An error occured...")
        print(e)
