# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import shutil
import os
import argparse

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="output path to serialize model")
ap.add_argument("-l", "--logs", required=True, help="tensorboard output logdirs")
args = vars(ap.parse_args())

print("[INFO] splitting dataset...")
imagePaths = list(paths.list_images(args["dataset"]))
(trainPaths, testPaths) = train_test_split(
    imagePaths, test_size=0.2, random_state=42)
(trainPaths, valPaths) = train_test_split(trainPaths, test_size=0.1, random_state=42)

trainPath = os.path.join(args["dataset"], "train")
valPath = os.path.join(args["dataset"], "val")
testPath = os.path.join(args["dataset"], "test")

datasets = [
    (trainPath, trainPaths),
    (valPath, valPaths),
    (testPath, testPaths)
]

for (dstPath, imagePaths) in datasets:
    if os.path.exists(dstPath):
        continue
    for imagePath in imagePaths:
        classDstPath = os.path.join(dstPath, imagePath.split(os.path.sep)[-2])
        if not os.path.exists(classDstPath):
            os.makedirs(classDstPath)
        shutil.copy(imagePath, classDstPath)

print("[INFO] data loading and augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    trainPath,
    target_size=(32, 32),
    batch_size=32,
    class_mode="categorical")

val_generator = test_datagen.flow_from_directory(
    valPath,
    target_size=(32, 32),
    batch_size=32,
    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(
    testPath,
    target_size=(32, 32),
    batch_size=32,
    class_mode="categorical")

print("[INFO] compiling model...")
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=3)
model.compile(optimizer=Adam(lr=0.005),
              loss="categorical_crossentropy", metrics=["accuracy"])
tensorboard = TensorBoard(log_dir=args["logs"])

print("[INFO] training the network...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    verbose=1,
    callbacks=[tensorboard])

print("[INFO] serializing model...")
model.save(args["model"])

print("[INFO] evaluating...")
preds = model.predict(test_generator)
print(classification_report(test_generator.argmax(axis=1), preds.argmax(axis=1), target_names=["cat", "dog", "panda"]))
