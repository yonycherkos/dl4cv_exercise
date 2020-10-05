# import the necessary packages
from tensorflow.keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

from pyimagesearch.preprocessing.simplePreprocessor import SimplePreprocessor
from pyimagesearch.preprocessing.imageToArrayPreprocessor import ImageToArrayPreprocessor
from pyimagesearch.datasets.simpleDatasetLoader import SimpleDatasetLoader

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="datasets/animals/",
                help="path to input dataset")
ap.add_argument("-m", "--model", default="output/models/shallownet_weights.hdf5",
                help="path to pre-trained model")
args = vars(ap.parse_args())

# load and preprocess test dataset
print("[INFO] loading dataset...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=-1)
data = data.astype("float")/255.0
classLabels = ["cat", "dog", "panda"]

print("[INFO] loading pre-trained model")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(data).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)
