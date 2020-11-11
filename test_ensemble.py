# import the necessary packages
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import os

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True, help="path to trained models")
args = vars(ap.parse_args())

print("[INFO] loading data...")
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
testY = lb.fit_transform(testY)

print("[INFO] loading models")
p = os.path.sep.join([args["models"], "*.hdf5"])
modelPaths = list(glob.glob(p))

models = []
for modelPath in modelPaths:
    model = load_model(modelPath)
    models.append(model)

print("[INFO] evaluating ensemble...")
predictions = []
for model in models:
    prediction = model.predict(testX, batch_size=32)
    predictions.append(prediction)

predictions = np.average(predictions, axis=0)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))