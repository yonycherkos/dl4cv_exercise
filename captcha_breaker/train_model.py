# import the necessary packages
import sys
sys.path.append("../")

import os
import cv2
import argparse
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import img_to_array
from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.utils.captchahelper import preprocess


# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="output path to serialize model")
ap.add_argument("-l", "--logs", required=True,
                help="output path to tensorboard logs")
args = vars(ap.parse_args())

print("[INFO] loading dataset...")
imagePaths = list(paths.list_images(args["dataset"]))

data = []
labels = []
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    label = imagePath.split(os.path.sep)[-2]
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

print("[INFO] splitting dataset...")
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling model...")
model = LeNet.build(28, 28, 1, 9)
opt = SGD(lr=0.01)
model.compile(optimizer=opt, loss="categorical_crossentropy",
              metrics=["accuracy"])

print("[INFO] training model...")
tensorboard = TensorBoard(log_dir=args["logs"])
model.fit(trainX, trainY, validation_data=(testX, testY),
          epochs=100, batch_size=32, callbacks=[tensorboard], verbose=1)

print("[INFO] evaluating model...")
preds = model.predict(testX)
print(classification_report(testY.argmax(axis=1),
                            preds.argmax(axis=1), target_names=lb.classes_))

print("[INFO] serializing model...")
model.save(args["model"])
