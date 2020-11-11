# import the necessary packages
from tensorflow.keras.applications.resnet50 import ResNet50
import argparse

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=-1, help="weither to include the top layer")
args = vars(ap.parse_args())

print("[INFO] loading network...")
model = ResNet50(weights="imagenet")

print("[INFO] showing layers...")
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))