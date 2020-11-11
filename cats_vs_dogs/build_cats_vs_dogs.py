# import the necessary packages
from sklearn.model_selection import train_test_split
from config import cats_vs_dogs_config as config
from imutils import paths
import numpy as np
import shutil
import json
import cv2
import os

# load and split images
print("[INFO] loading data ...")
imagePaths = list(paths.list_images(config.IMAGES_PATH))

# split the dataset
(trainPaths, testPaths) = train_test_split(imagePaths, test_size=config.TEST_SPLIT, random_state=42)
(trainPaths, valPaths) = train_test_split(trainPaths, test_size=config.VAL_SPLIT, random_state=42)

datasets = [
    ("train", trainPaths, config.TRAIN_PATH),
    ("val", valPaths, config.VAL_PATH),
    ("test", testPaths, config.TEST_PATH)
]


(R, G, B) = ([], [], [])
# loop over the dataset tuples
for (dType, imagePaths, outputPath) in datasets:
    print("[INFO] building {} ...".format(outputPath))

    # loop over the imagePaths
    for (i, imagePath) in enumerate(imagePaths):
        # image = cv2.imread(imagePath)

        # # if we are building train dataset, comput mean for each channel
        # # and update the list
        # if dType == "train":
        #     (r, g, b) = cv2.mean(image)[:3]
        #     R.append(np.mean(r))
        #     G.append(np.mean(g))
        #     B.append(np.mean(b))

        try:
            classOutputPath = os.path.sep.join([outputPath, imagePath.split(os.path.sep)[-2]])
            if not os.path.exists(classOutputPath):
                os.makedirs(classOutputPath)
            shutil.copy(imagePath, classOutputPath)
        except:
            raise "Error copying {} imagePath to {}".format(imagePath, classOutputPath)

# # construct average of RGB mean, serialize it to json file.
# mean = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
# print("[INFO] serializing mean ...")
# f = open(config.DATASET_MEAN, "w")
# f.write(json.dumps(mean))
# f.close()