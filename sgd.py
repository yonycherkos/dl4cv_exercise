# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid_activation(x):
    return 1.0/(1 + np.exp(-x))

def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0] = 0
    preds[preds > 0] = 1
    return preds

def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch_size", type=str, default=32, help="size of SGD mini-batch")
args = vars(ap.parse_args())

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5)
y = y.reshape((y.shape[0], 1))
X = np.c_[X, np.ones((X.shape[0]))]
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.25, random_state=42)

print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)

losses = []
for epoch in np.arange(0, args["epochs"]):
    epochLosses = []
    for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):
        preds = predict(batchX, W)
        error = preds - batchY
        epochLosses.append(np.sum(error**2))

        gradient = batchX.T.dot(error)
        W += -args["alpha"] * gradient

    loss = np.average(epochLosses)
    losses.append(loss)
    if epoch == 0 or (epoch % 5 == 0):
        print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
colors = []
for y in testY:
    if y == 0:
        colors.append("b")
    else:
        colors.append("r")
plt.style.use("ggplot")
plt.figure()
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=colors, s=30)
plt.title("Data")

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()