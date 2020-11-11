# import the necessary packages
import sys
sys.path.append("../")

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from pyimagesearch.io.hdf5DatasetWriter import Hdf5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import progressbar
import argparse
import cv2
import os

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset.")
ap.add_argument("-o", "--outut", required=True, help="output path to store the generated hdf5 file.")
ap.add_argument("-b", "--batch-size", required=True, help="# of batchs image to process.")
ap.add_argument("-s", "--buffer-size", requried=True, help="memory buffer-size")
args = vars(ap.parse_args())

# construct input image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
np.random.shuffle(imagePaths)

# extract and encode labels
labels = [imagePath.split(os.path.sep)[-2] for imagePath in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load model
print("[INFO] loading network...")
model = VGG16(weight="imagenet", include_top=False)

# initialize hdf5 dataset write, then store class label
dataset = Hdf5DatasetWriter((len(imagePaths), 512 * 7 * 7), args["output"], dataKey="features", buffer_size=args["buffer_size"])
dataset.storeClassLabels(le.classes_)

# construct progress bar
bs = args["batch_size"]
widgets = ["Extract Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths)//bs, widgets=widgets)

# loop over the imagePaths
for i in np.arange(0, imagePaths, bs):
    # read and preprocess the image
    batchImagesPath = imagePaths[i:i + bs]
    batchImages = []
    batchLabels = labels[i:i + bs]
    for imagePath in batchImagesPath:
        image = load_img(imagePath, target_size=(244, 244))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        batchImages.append(image)

    # make prediction on batch images
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # add the batch of features to hdf5
    dataset.add(features, batchLabels)
    pbar.update(i)

# close hdf5 writer
dataset.close()
pbar.finish()