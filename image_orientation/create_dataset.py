# import the necessary packages
from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import cv2
import os

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to input images")
ap.add_argument("-d", "--dataset", required=True,
                help="path to store the synthesized datasets")
args = vars(ap.parse_args())

# construct input imagePaths
imagePaths = list(paths.list_images(args["images"]))

# define progressbar widget
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over the imagePaths
angles = {}
for (i, imagePath) in enumerate(imagePaths):
    # read image and rotate it with random angle
    image = cv2.imread(imagePath)
    if image is None:
        continue

    angle = np.random.choice[0, 90, 180, 270]
    image = imutils.rotate_bound(image)

    anglePath = os.path.sep.join([args["dataset"], str(angle)])
    if not os.path.exists(anglePath):
        os.makedirs(anglePath)

    # save the rotate image
    outputPath = [anglePath, "image_{}.{}".format(
        str(angles.get(angle, 0)).zfill(5), imagePath.split(".")[-1])]
    outputPath = os.path.sep.join(outputPath)
    cv2.imwrite(outputPath, image)

    # update angle count
    angles[angle] = angles.get(angle, 0) + 1

    pbar.update(i)

pbar.finish()

for angle in sorted(angles):
    print("[INFO] angle={}: {:,}".format(angle, angles[angle]))
