# import the necessary packages
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pyimagesearch.io.hdf5DatasetWriter import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from imutils import paths
import numpy as np
import progressbar
import argparse
import os

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output the hdf5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32, help="batch-size of images")
ap.add_argument("-s", "--buffer-size", type=int, default=1000, help="size of feature extract buffer")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
np.random.shuffle(imagePaths)

labels = [imagePath.split(os.path.sep)[-2] for imagePath in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading the model...")
resnet_model = ResNet50(weights="imagenet")
model = Model(inputs=resnet_model.input,
              outputs=resnet_model.layers[-3].output)

dataset = HDF5DatasetWriter(dims=(len(imagePaths), 7 * 7 * 2048),
                            outputPath=args["output"], dataKey="features", bufferSize=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

bs = args["batch_size"]
for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabels = labels[i:i + bs]

    batchImages = []
    for imagePath in batchPaths:
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 7 * 7 * 2048))

    dataset.add(features, batchLabels)
    pbar.update(i)

dataset.close()
pbar.finish()
