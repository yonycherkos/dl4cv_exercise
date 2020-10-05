# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def convolution(image, K):
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]
    pad = (kW - 1)//2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]
            k = (roi*K).sum()
            output[y - pad, x - pad] = k
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    return output


# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image")
args = vars(ap.parse_args())

# define kernels to apply
smallBlur = np.ones((7, 7), dtype="float")*(1.0/(7*7))
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
output = convolution(gray, smallBlur)
cv2.imshow("image", image)
cv2.imshow("gray", gray)
cv2.imshow("output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
