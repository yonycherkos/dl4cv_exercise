# import the necessary packages
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from pyimagesearch.preprocessing.imageToArrayPreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.aspectAwarePreprocessor import AspectAwarePreprocessor
from pyimagesearch.datasets.simpleDatasetLoader import SimpleDatasetLoader
from pyimagesearch.nn.conv.minigooglenet import MiniGoogleNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
import argparse
import os

# define initiale learning rate and training epochs
INIT_LR = 1e-3
NUM_EPOCHS = 70


def poly_decay(epoch):
    # initialize base learning rate, maximum epochs and power terms.
    baseLR = INIT_LR
    maxEpoch = NUM_EPOCHS
    power = 1.0

    # compute new learning rate
    alpha = baseLR + (1 - epoch/maxEpoch)**power

    # return the compute new lerning rate
    return alpha


# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="datasets/animals", help="path to input dataset")
ap.add_argument("-m", "--model", default="output/models/minigooglenet_animals.hdf5",
                help="path to save trained model")
ap.add_argument("-l", "--logs", default="output/logs/miniGoogleNet",
                help="path to output tensorboard logs")
ap.add_argument("-r", "--report", default="output/minigooglenet_animals_report.txt", help="path to store evalution report")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor(data_format="channels_last")
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define train and validation data augmentation
trainAug = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
                              height_shift_range=0.2, shear_range=0.2, zoom_range=0.2)

# define learning rate scheduler, and tensorboard callbacks
callbacks = [LearningRateScheduler(
    poly_decay), TensorBoard(log_dir=args["logs"])]

print("[INFO] compiling model...")
model = MiniGoogleNet.build(height=224, width=224, depth=3, classes=2)
model.compile(Adam(lr=INIT_LR), loss="categorical_crossentropy",
              metrics=["accuracy"])

print("[INFO] training model...")
model.fit(trainAug.flow(trainX, trainY), validation_data=(testX, testY), batch_size=32, epochs=NUM_EPOCHS, verbose=1, callbacks=callbacks)

print("[INFO] Evaluatin model...")
report = classification_report(testY.argmax(axis=1), model.predict(testX).argmax(axis=1), target_names=["cats" "dogs"])
f = open(args["report"], "w")
f.write(report)
f.close()

print("[INFO] serializing trained model...")
model.save(args["model"])
