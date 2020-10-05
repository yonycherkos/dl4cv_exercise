# import the necessary packages
from pyimagesearch.nn.neuralNetwork import NeuralNetwork
import numpy as np

# construct XOR dataset")
X = np.array([[0, 0], [0, 1], [1,  0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(layers=[2, 2, 1])
nn.fit(X, y, epochs=20000)

print("[INFO] evaluating...")
for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={}, stpe={}".format(x, target, pred, step))