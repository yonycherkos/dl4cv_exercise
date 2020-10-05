# import the necessary packages
import sys
sys.path.append("../")

from pyimagesearch.utils.captchahelper import preprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.contours import sort_contours
from imutils import grab_contours
from imutils import paths
import numpy as np
import argparse
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input directory of images")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained model...")
model = load_model(args["model"])

print("[INFO] loading input images...")
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,), replace=False)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4] # sort contours by contour size
    cnts = sort_contours(cnts)[0] # sort contours from left-to-right

    output = cv2.merge([gray]*3)
    predictions = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

        roi = preprocess(roi, 28, 28)
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = roi / 255.0

        pred = model.predict(roi).argmax(axis=1)[0] + 1
        cv2.rectangle(output, (x - 3, y - 3), (x + w + 3, y + h + 3), (0, 255, 0), 2)
        cv2.putText(output, str(pred), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        predictions.append(str(pred))

    print("[INFO] captcha: {}".format("".join(predictions)))
    cv2.imshow("Captcha", output)
    cv2.waitKey(0)