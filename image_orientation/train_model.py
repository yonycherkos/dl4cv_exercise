# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np
import argparse
import pickle
import h5py

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to hdf5 database.")
ap.add_argument("-m", "--model", required=True, help="output path to store trained model.")
ap.add_argument("-j", "--jobs", default=-1, type=int, help="# jobs to use")
args = vars(ap.parse_args())

# load hdf5 database
print("[INFO] loading hdf5 database...")
db = h5py.File(args["db"])
(trainX, testX, trainY, testY) = train_test_split(db["features"], db["labels"], test_size=0.25, random_state=42)

# train logistic regression
print("[INFO] tuning hyperparametries...")
parms = {"C": [0.1, 1.0, 10, 100, 1000, 10000]}
model = GridSearchCV(LogisticRegression, parms, cv=3, n_jobs=args["jobs"])
model.fit(trainX, trainY)
print("[INFO] best hyperparmetries: {}".format(model.best_params_))

# evaluating the model performance
print("[INFO] evaluating model...")
print(classification_report(testY.argmax(axis=1), model.predict(testX).argmax(axis=1), target_names=db["class_labels"]))

# save the model
print("[INFO] saving the model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()