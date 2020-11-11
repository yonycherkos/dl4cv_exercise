# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import argparse
import pickle
import h5py

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to input hdf5 dataset")
ap.add_argument("-m", "--model", required=True, help="output path to serialize model")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs to use when tuning hyperparametries")
args = vars(ap.parse_args())

print("[INFO] loading hdf5 dataset...")
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[INFO] tuning hyperparametries...")
parms = {"C": [0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), parms, cv=3, n_jobs=args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameter: {}".format(model.best_params_))

print("[INFO] evaluting...")
print(classification_report(db["labels"][i:], model.predict(db["features"][i:]), target_names=db["label_names"]))

print("[INFO] serializing model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()