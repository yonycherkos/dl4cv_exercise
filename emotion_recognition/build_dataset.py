# import the necessry packages
from config import emotion_config as config
import numpy as np
import cv2
import os

print("[INFO] loading input data...")
f = open(config.INPUT_PATH)

# define train, val, and test lists
(trainImages, trainLabels) = ([], []) 
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

# loop over the rows of input data
for row in f:
    (label, image, usage) = row.strip().split(",")
    image = np.array(image, dtype="uint8")
    image = image.reshape((48, 48))

    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)
    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)
    else:
        testImages.append(image)
        testLabels.append(label)

# define a list of tuples of train, val, and test
datasets = [
    (trainImages, trainLabels, config.TRAIN_PATH),
    (valImages, valLabels, config.VAL_PATH),
    (testImages, testLabels, config.TEST_PATH)
]

for (images, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    for (image, label) in zip(images, labels):
        classOutputPath = os.path.sep.join([outputPath, label])
        if not os.path.exists(classOutputPath):
            os.makedirs(classOutputPath)
        cv2.imwrite(classOutputPath, image)