# import the necessary packages
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import argparse
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# step - 1: laoding pre-trained model
print("[INFO] loading pre-trained model...")
model = ResNet50(weights="imagenet")

# step - 2: loading and pre-processing input image
print("[INFO] loading and pre-processing input image...")
inputShape = (224, 224)
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)

# step - 3: classifying image...
print("[INFO] classifying image..")
preds = model.predict(image)
P = decode_predictions(preds)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}".format(i + 1, label, prob*100))

orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)