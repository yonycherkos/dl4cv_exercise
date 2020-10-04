# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.datasets.simpleDatasetLoader import SimpleDatasetLoader
from pyimagesearch.preprocessing.simplePreprocessor import SimplePreprocessor
from imutils import paths
import numpy as np
import argparse

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", help="path to image datasets")
ap.add_argument("-n", "--neighbors", default=1,
                help="number of closest votes to consider")
ap.add_argument("-j", "--jobs", default=-1,
                help="number of processer or core to use (-1 means use of cores)")
args = vars(ap.parse_args())

# step - 1: gather the dataset(load and preprocess dataset)
print("[INFO] loading dataset...")
imagePaths = list(paths.list_images("datasets/animals/"))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)

data = data.reshape((data.shape[0], 3072))
print("[INFO] dataset size: {:.2f}MB".format(data.nbytes/(1024 * 1000)))
le = LabelEncoder()
labels = le.fit_transform(labels)

# step - 2: split the dataset
print("[INFO] spliting the dataset...")
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42)

# step - 3: train the classifier
print("[INFO] training the classifier...")
model = KNeighborsClassifier(
    n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)

# step - 4: evaluate
print("[INFO] evaluation...")
print(classification_report(testY, model.predict(testX)))
