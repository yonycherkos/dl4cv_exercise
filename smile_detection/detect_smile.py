# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import argparse
import time
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained smile detector model")
ap.add_argument("-c", "--cascade", required=True,
                help="path to haarcascade face detector model")
ap.add_argument("-v", "--video", type=str, default=None,
                help="path to optional video file")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained model...")
model = load_model(args["model"])

print("[INFO] load haarcascade face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

if not args.get("video"):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    rects = detector.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        cv2.putText(frameClone, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0))
        cv2.rectangle(frameClone, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face", frameClone)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
