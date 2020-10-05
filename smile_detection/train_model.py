# import the necessary packages
import sys
sys.path.append("../")

from pyimagesearch.nn.conv.lenet import LeNet
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import imutils
import numpy as np
import argparse
import cv2
import os

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="input path to directory of images.")
ap.add_argument("-m", "--model", required=True, help="output path to serialize model")
ap.add_argument("-l", "--logs", required=True, help="output path to tensorboard logs")
ap.add_argument("-r", "--report", required=True, help="output file to store classification report")
args = vars(ap.parse_args())

print("[INFO] loading and preprocessing dataset...")
imagePaths = list(paths.list_images(args["dataset"]))

data = []
labels = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

classTotal = labels.sum(axis=0)
classWeight = classTotal.max() / classTotal

print("[INFO] splitting dataset...")
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

print("[INFO] training model...")
tensorboard = TensorBoard(log_dir=args["logs"])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=15, batch_size=32, class_weight=classWeight, callbacks=[tensorboard], verbose=1)

print("[INFO] evaluating model...")
preds = model.predict(testX)
report = classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=le.classes_)

print("[INFO] saving classification report..")
with open(args["report"], "w") as f:
    f.write(report)
print(report)

print("[INFO] serializing model...")
model.save(args["model"])