# import the necessary packages
from pyimagesearch.utils.ranked import rank5_accuracy
import argparse
import pickle
import h5py

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to input hdf5 dataset")
ap.add_argument("-m", "--model", required=True, help="path to serialized pre-trained model")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained model...")
model = pickle.loads(open(args["model"], "rb").read())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[INFO] predicting...")
preds = model.predict(db["features"][1:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i])
print("[INFO] rank1: {}".format(rank1))
print("[INFO] rank5: {}".format(rank5))

db.close()