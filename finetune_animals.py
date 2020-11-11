# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.nn.conv.fchead import FCHeadNet
from imutils import paths
import numpy as np
import argparse
import os

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to serialize model")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, fill_mode="nearest")

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

data = []
labels = []
for imagePath in imagePaths:
    image = load_img(imagePath, (224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    label = imagePath.split(os.path.sep)[-1]
    data.append(image)
    labels.append(label)
    
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] building model...")
baseModel = ResNet50(weights="imagenet", include_top=False)
headModel = FCHeadNet.build(baseModel, len(le.classes_), 256)
model = Model(inputs=baseModel.input, outputs=headModel.output)

# freeze the base conv layers
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] training model...")
model.fit(aug.flow(trainX,trainY, batch_size=32), batch_size=32, epochs=15, verbose=1, validation_data=(testX, testY))

print("[INFO] evaluating...")
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

# unfreeze the last conv layers
for layer in baseModel[:-5]:
    layer.trainable = True

print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] re-training model...")
model.fit(aug.flow(trainX,trainY, batch_size=32), batch_size=32, epochs=15, verbose=1, validation_data=(testX, testY))

print("[INFO] re-evaluating...")
print(classification_report(testY, model.predict(testX), target_names=le.classes_))

print("[INFO] saving model...")
model.save(args["model"])