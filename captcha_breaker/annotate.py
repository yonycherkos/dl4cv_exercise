# import the necessary packages
from imutils import grab_contours
from imutils import paths
import imutils
import argparse
import cv2
import os

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the donwloaded input images")
ap.add_argument("-a", "--annot", required=True, help="path to output annotated images")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))
counts = {}
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    try:
        # read image, covert it to gray scale, then apply padding
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the gray scale image, and find contours
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        # loop over the contours
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)
            if chr(key) == "'":
                print("[INFO] ignoring character...")
                continue

            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            counts[key] = count + 1
    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break
    
    except:
        print("[INFO] skipping image...")