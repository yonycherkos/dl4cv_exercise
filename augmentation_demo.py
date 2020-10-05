# import the necessary packages
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-o", "--output", required=True,
                help="output path to store generated images")
ap.add_argument("-p", "--prefix", type=str, default="image",
                help="output filename prefix")
args = vars(ap.parse_args())

print("[INFO] loading and preprocessing sample image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] generating images...")
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest")

datagen = aug.flow(image, batch_size=1,
                   save_to_dir=args["output"], save_prefix=args["prefix"], save_format="jpg")

total = 0
for image in datagen:
    total += 1
    if total == 10:
        break

print("[INFO] loading generated images...")
imagePaths = list(paths.list_images(args["output"]))
images = []
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    images.append(image)

print("[INFO] building montages...")
montages = build_montages(images, (224, 224), (5, 2))
for montage in montages:
    cv2.imshow("Montage", montage)
    cv2.waitKey(0)