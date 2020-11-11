# import the necessary packages
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import h5py
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to hdf5 database")
ap.add_argument("-i", "--dataset", required=True, help="path to input image dataset")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())


# load sample images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

# load the models
print("[INFO] loading the models...")
vgg = VGG16(weight="imagenet", include_top=False)
model = pickle.loads(open(args["model"], "rb").read())

# load the label_names
db = h5py.File(args["db"])
labelNames = [int(label) for label in db["label_names"][:]]
db.close()

# loop over the sampel image paths
for imagePath in imagePaths:
    orig = cv2.imread(imagePath)

    # read and preprocess images
    image = load_img(imagePath, target_size=(244, 244))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # make prediction on the preprocessed image
    features = vgg.predict(image)
    preds = model.predict(features)
    angle = labelNames[preds[0]]

    # rotate the image with the predict angle
    rotated = imutils.rotate_bound(image, 360 - angle)

    # dispaly the rotated and original image
    cv2.imshow("Original", orig)
    cv2.imshow("Corrected", rotated)
    cv2.waitKey(0)