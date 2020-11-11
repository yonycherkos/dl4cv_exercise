# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where face cascade detector reside.")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained emotion classifier model")
ap.add_argument("-v", "--video", default=None, type=str,
                help="path to optional video file")
args = vars(ap.parse_args())

# load face detectore and pre-trained model
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["happy", "angry"]

# if video file not supplied, start the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# else load the video file
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

    if args.get("video", False) and grabbed is None:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, miniNeighbors=5, minSize=(
        30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) > 0:
        rect = sorted(rects, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)[0]
        (fX, fY, fW, fH) = rect

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float")/255.
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        for (i, (emotion, prob)) in zip(EMOTIONS, preds):
            text = "{}: {:.2f}".format(emotion, prob * 100)
            w = int(prob * 300)

            cv2.rectangle(canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        cv2.rectangle(frameClone, (fX, fY), (fX + w, fY + fH), (0, 0, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Emotion", frameClone)
    cv2.imshow("Probabilities", canvas)

camera.release()
cv2.destroyAllWindows()