# import the necessary packages
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse

from pyimagesearch.datasets.simpleDatasetLoader import SimpleDatasetLoader
from pyimagesearch.preprocessing.simplePreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.imageToArrayPreprocessor import ImageToArrayPreprocessor
from pyimagesearch.nn.conv.shallownet import ShallowNet

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", requried=True, help="path to animal dataset.")
ap.add_argument("-m", "--model", required=True,help="path to output serialized model")
ap.add_argument("-l", "--logs", required=True,help="path to tensorboard log-dir")
args = vars(ap.parse_args())

# step - 1: gather dataset(load and preprocess)
print("[INFO] loading data...")
imagePaths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor(data_format="channels_last")
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

# step - 2: split the dataset
print("[INFO] splitting dataset...")
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# step - 3: train
print("[INFO] compiling model...")
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
opt = SGD(lr=0.005)
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
tensorboard = TensorBoard(log_dir=args["logs"])
model.fit(trainX, trainY, validation_data=(testX, testY),
          batch_size=32, epochs=100, verbose=1, callbacks=[tensorboard])

print("[INFO] serializing model...")
model.save(args["model"])

# step - 4: evaluate
print("[INFO] evaluating network...")
preds = model.predict(testX)
print(classification_report(testY.argmax(axis=1), preds.argmax(
    axis=1), target_names=["cat", "dog", "panda"]))
