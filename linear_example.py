# import the necessary packages
import numpy as np
import cv2

# initialize the weight and bias
labels = ["dog", "cat", "panda"]
np.random.seed(1)
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# load the input image
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

# compute score function
scores = np.dot(W, image) + b
for (label, score) in zip(labels, scores):
    print("{}: {}".format(label, score))

cv2.putText(orig, "Label: {}".format(labels[np.argmax(
    scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("orig", orig)
cv2.waitKey()
cv2.destroyAllWindows()
